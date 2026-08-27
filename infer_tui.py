# infer_tui.py
"""ETET TUI Chat: Textual-based chat interface for MiniCPM5 / ETET models."""
from __future__ import annotations

import argparse
import base64
import copy
import gc
import json
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image as PILImage
from rich.markup import escape
from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    SiglipVisionModel,
)

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Input, Static, TextArea

# ============================================================
# ETET configuration (must match upcycle.py)
# ============================================================
NUM_EXPERTS = 3
ROUTER_TOP_K = 1
MOE_START_LAYER = 16
MOE_END_LAYER = 24
NUM_TOTAL_LAYERS = 24
IMAGE_SIZE = 512
IMAGE_TOKEN = "<image>"

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"


def is_etet_model(model_dir: Path) -> bool:
    return "etet" in model_dir.name.lower()


def is_multimodal_model(model_dir: Path) -> bool:
    name = model_dir.name.lower()
    return "etet" in name and "vl" in name


def find_models() -> List[Path]:
    if not MODELS_DIR.exists():
        return []
    return sorted(
        [p for p in MODELS_DIR.iterdir() if p.is_dir() and p.name.lower() not in ("tmp", "gguf")],
        key=lambda p: p.name.lower(),
    )


def format_params(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return str(value)


def validate_config(config) -> None:
    model_type = getattr(config, "model_type", None)
    num_layers = getattr(config, "num_hidden_layers", None)
    if model_type != "llama":
        raise ValueError(f"Expected model_type='llama', got {model_type!r}")
    if num_layers != NUM_TOTAL_LAYERS:
        raise ValueError(f"Expected {NUM_TOTAL_LAYERS} layers, got {num_layers}")


def split_think(text: str) -> Tuple[str, str]:
    # Split a model response into (think_content, main_content).
    m = re.search(re.escape(THINK_OPEN) + r"(.*?)" + re.escape(THINK_CLOSE), text, re.DOTALL)
    if m:
        think = m.group(1).strip()
        main = (text[:m.start()] + text[m.end():]).strip()
        return think, main
    idx = text.find(THINK_CLOSE)
    if idx != -1:
        # Template pre-opened <think>, model output starts inside the think block.
        return text[:idx].strip(), text[idx + len(THINK_CLOSE):].strip()
    idx = text.find(THINK_OPEN)
    if idx != -1:
        # Unterminated think block (truncated generation).
        return text[idx + len(THINK_OPEN):].strip(), text[:idx].strip()
    return "", text.strip()


def strip_think_tags(text: str) -> str:
    # Remove think markers from history messages before applying chat template.
    think, main = split_think(text)
    return main if main else text


def copy_via_osc52(text: str) -> None:
    # Copy text to the system clipboard using the OSC 52 escape sequence.
    data = base64.b64encode(text.encode("utf-8")).decode("ascii")
    sys.stdout.write(f"\033]52;c;{data}\a")
    sys.stdout.flush()


# ============================================================
# ETET MoE modules (must match upcycle.py structure)
# ============================================================
class Top1Router(nn.Module):
    """Top-1 router for MoE with automatic dtype casting."""

    def __init__(self, hidden_size: int, num_experts: int, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = ROUTER_TOP_K
        self.linear = nn.Linear(hidden_size, num_experts, bias=False, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor):
        original_shape = hidden_states.shape
        hidden_size = original_shape[-1]
        flat_states = hidden_states.reshape(-1, hidden_size)
        if self.linear.weight.dtype != flat_states.dtype:
            flat_states = flat_states.to(self.linear.weight.dtype)
        router_logits = self.linear(flat_states)
        router_probs = torch.softmax(router_logits.float(), dim=-1).to(router_logits.dtype)
        topk_probs, topk_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        return (
            router_logits.reshape(*original_shape[:-1], self.num_experts),
            topk_probs.reshape(*original_shape[:-1], self.top_k),
            topk_indices.reshape(*original_shape[:-1], self.top_k),
        )


class ETETMoE(nn.Module):
    """MoE module replacing the dense FFN: 3 experts, Top-1 routing."""

    def __init__(self, dense_ffn: nn.Module, hidden_size: int,
                 num_experts: int = NUM_EXPERTS, top_k: int = ROUTER_TOP_K):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        router_dtype = next(dense_ffn.parameters()).dtype
        self.router = Top1Router(hidden_size, num_experts, dtype=router_dtype)
        self.experts = nn.ModuleList([copy.deepcopy(dense_ffn) for _ in range(num_experts)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _, router_probs, router_indices = self.router(hidden_states)
        original_shape = hidden_states.shape
        hidden_size = original_shape[-1]
        flat_states = hidden_states.reshape(-1, hidden_size)
        flat_indices = router_indices.reshape(-1)
        flat_probs = router_probs.reshape(-1, self.top_k)
        output = torch.zeros_like(flat_states)
        for expert_idx in range(self.num_experts):
            token_indices = (flat_indices == expert_idx).nonzero(as_tuple=False).flatten()
            if token_indices.numel() == 0:
                continue
            expert_input = flat_states.index_select(0, token_indices)
            expert_output = self.experts[expert_idx](expert_input)
            if isinstance(expert_output, tuple):
                expert_output = expert_output[0]
            weights = flat_probs.index_select(0, token_indices)[:, 0]
            expert_output = expert_output * weights.unsqueeze(-1)
            output.index_add_(0, token_indices, expert_output)
        return output.reshape(*original_shape[:-1], hidden_size)


# ============================================================
# Checkpoint loading
# ============================================================
def find_checkpoint_files(model_dir: Path) -> List[Path]:
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        shard_names = sorted(set(weight_map.values()))
        files = [model_dir / name for name in shard_names]
        missing = [p for p in files if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing checkpoint shards:\n" + "\n".join(str(x) for x in missing)
            )
        return files
    single_file = model_dir / "model.safetensors"
    if single_file.exists():
        return [single_file]
    files = sorted(model_dir.glob("*.safetensors"))
    if files:
        return files
    raise FileNotFoundError(f"No safetensors checkpoint found in: {model_dir}")


def load_safetensors_into_model(model: nn.Module, model_dir: Path,
                                 log_fn: Optional[Callable[[str], None]] = None) -> None:
    def log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    checkpoint_files = find_checkpoint_files(model_dir)
    log(f"Found {len(checkpoint_files)} safetensors shard(s).")
    model_state_keys = set(model.state_dict().keys())
    loaded_keys = set()
    unexpected_keys = set()
    for idx, checkpoint_file in enumerate(checkpoint_files, start=1):
        log(f"Loading shard {idx}/{len(checkpoint_files)}: {checkpoint_file.name}")
        shard_state = load_file(str(checkpoint_file), device="cpu")
        shard_keys = set(shard_state.keys())
        unexpected_keys.update(shard_keys - model_state_keys)
        loaded_keys.update(shard_keys & model_state_keys)
        model.load_state_dict(shard_state, strict=False)
        del shard_state
        gc.collect()
    missing_keys = model_state_keys - loaded_keys
    if missing_keys:
        preview = sorted(missing_keys)[:30]
        raise RuntimeError(
            f"Checkpoint missing {len(missing_keys)} tensors.\n" + "\n".join(preview)
        )
    if unexpected_keys:
        preview = sorted(unexpected_keys)[:30]
        raise RuntimeError(
            f"Checkpoint has {len(unexpected_keys)} unexpected tensors.\n" + "\n".join(preview)
        )
    log("Weights loaded successfully.")


# ============================================================
# Multimodal components (VL models only)
# ============================================================
class ETETVisionConnector(nn.Module):
    """Projects SigLIP-HD features into the language model space."""

    def __init__(self, vision_hidden_size: int, lm_hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vision_hidden_size, lm_hidden_size),
            nn.GELU(),
            nn.Linear(lm_hidden_size, lm_hidden_size),
            nn.LayerNorm(lm_hidden_size, eps=1e-6),
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        return self.net(image_features)


class ETETVisionTower(nn.Module):
    """SigLIP-HD wrapper with 512x512 interpolated position encoding."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.expected_patches = ((IMAGE_SIZE - 14) // 14 + 1) ** 2

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        try:
            output = self.model(pixel_values=pixel_values, interpolate_pos_encoding=True)
        except TypeError:
            output = self.model(pixel_values=pixel_values)
        hidden = output.last_hidden_state
        if hidden.dim() == 3 and hidden.shape[1] == self.expected_patches + 1:
            hidden = hidden[:, 1:]
        return hidden


class MultimodalComponents:
    def __init__(self, vision_tower, connector, image_token_id, processor):
        self.vision_tower = vision_tower
        self.connector = connector
        self.image_token_id = image_token_id
        self.processor = processor


def load_multimodal_components(model_dir: Path, model: nn.Module, tokenizer,
                               dtype: torch.dtype, device: torch.device,
                               log_fn: Optional[Callable[[str], None]] = None):
    def log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    vision_dir = model_dir / "vision_tower"
    if not vision_dir.exists():
        raise FileNotFoundError(f"Vision tower directory not found: {vision_dir}")

    log("Loading SigLIP-HD vision encoder...")
    tower_model = SiglipVisionModel.from_pretrained(
        vision_dir, local_files_only=True, torch_dtype=dtype
    )
    vision_tower = ETETVisionTower(tower_model)
    vision_tower.to(device=device, dtype=dtype)
    vision_tower.eval()

    log("Loading vision connector...")
    connector_dir = model_dir / "vision_connector"
    connector_state = load_file(str(connector_dir / "connector.safetensors"))
    vision_hidden = tower_model.config.hidden_size
    lm_hidden = model.config.hidden_size
    connector = ETETVisionConnector(vision_hidden, lm_hidden)
    connector.load_state_dict(connector_state)
    connector.to(device=device, dtype=dtype)
    connector.eval()

    log("Loading image processor...")
    processor = AutoImageProcessor.from_pretrained(vision_dir, local_files_only=True)
    processor.size = {"height": IMAGE_SIZE, "width": IMAGE_SIZE}

    image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    if not isinstance(image_token_id, int) or image_token_id < 0:
        raise ValueError(f"Image token {IMAGE_TOKEN} not found in tokenizer.")
    log(f"Image token ID: {image_token_id}")
    return MultimodalComponents(vision_tower, connector, image_token_id, processor)


# ============================================================
# Generation
# ============================================================
def get_stop_token_ids(tokenizer) -> List[int]:
    ids: List[int] = []
    try:
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_id, int) and im_end_id >= 0:
            ids.append(im_end_id)
    except Exception:
        pass
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
    return list(dict.fromkeys(ids))


def _locate_image_anchor(ids_list: List[int], image_token_id: int, tokenizer) -> Optional[int]:
    if image_token_id in ids_list:
        return ids_list.index(image_token_id)
    try:
        frag_ids = tokenizer.encode(IMAGE_TOKEN, add_special_tokens=False)
        if isinstance(frag_ids, int):
            frag_ids = [frag_ids]
        if frag_ids:
            n_frag = len(frag_ids)
            for start in range(len(ids_list) - n_frag + 1):
                if ids_list[start:start + n_frag] == frag_ids:
                    return start
    except Exception:
        pass
    return None


def apply_template_with_thinking(tokenizer, messages: List[Dict[str, str]]):
    # Apply chat template with thinking enabled; fall back if unsupported.
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True, enable_thinking=True,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        )


def run_generation(model, tokenizer, messages, device, max_new_tokens,
                   do_sample, temperature, top_p,
                   image: Optional[PILImage.Image] = None,
                   mm_components: Optional[MultimodalComponents] = None):
    messages_local = copy.deepcopy(messages)
    # Strip think blocks from history so the template stays clean,
    # while the local conversation keeps full think text for display/export.
    for m in messages_local:
        if m["role"] == "assistant":
            m["content"] = strip_think_tags(m["content"])

    use_vision = image is not None and mm_components is not None
    if use_vision:
        for i in reversed(range(len(messages_local))):
            if messages_local[i]["role"] == "user":
                messages_local[i]["content"] = IMAGE_TOKEN + "\n" + messages_local[i]["content"]
                break

    inputs = apply_template_with_thinking(tokenizer, messages_local)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(device)
    input_length = input_ids.shape[1]

    stop_ids = get_stop_token_ids(tokenizer)
    generation_kwargs: Dict = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "use_cache": True,
        "repetition_penalty": 1.15,
    }
    if stop_ids:
        generation_kwargs["eos_token_id"] = stop_ids if len(stop_ids) > 1 else stop_ids[0]
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    used_embeds = False
    if use_vision:
        pixel_values = mm_components.processor(
            images=image.convert("RGB"), return_tensors="pt"
        )["pixel_values"].to(next(mm_components.vision_tower.parameters()).device)
        with torch.no_grad():
            image_features = mm_components.vision_tower(pixel_values)
            projected = mm_components.connector(image_features)
        embed_layer = model.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids).to(device)
        ids_list = input_ids[0].tolist()
        anchor_pos = _locate_image_anchor(ids_list, mm_components.image_token_id, tokenizer)
        if anchor_pos is not None:
            num_visual = projected.shape[1]
            before = inputs_embeds[0, :anchor_pos]
            after = inputs_embeds[0, anchor_pos + 1:]
            visual_embeds = projected[0].to(inputs_embeds.dtype)
            inputs_embeds = torch.cat([before, visual_embeds, after], dim=0).unsqueeze(0)
            new_len = inputs_embeds.shape[1]
            attention_mask = torch.cat([
                attention_mask[0, :anchor_pos],
                torch.ones(num_visual, dtype=attention_mask.dtype, device=attention_mask.device),
                attention_mask[0, anchor_pos + 1:],
            ], dim=0).unsqueeze(0)
            generation_kwargs["inputs_embeds"] = inputs_embeds
            generation_kwargs["attention_mask"] = attention_mask
            input_length = new_len
            used_embeds = True
        else:
            generation_kwargs["input_ids"] = input_ids
            generation_kwargs["attention_mask"] = attention_mask
    else:
        generation_kwargs["input_ids"] = input_ids
        generation_kwargs["attention_mask"] = attention_mask

    start_time = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**generation_kwargs)
    elapsed = time.perf_counter() - start_time

    if used_embeds and generated.shape[-1] <= input_length:
        output_ids = generated
    else:
        output_ids = generated[:, input_length:]
    new_tokens = int(output_ids.shape[-1])
    # Keep <think>...</think> intact: skip_special_tokens=False and the
    # cleanup list below never touches the think markers.
    raw = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    for tok in ("<s>", "</s>", "<|im_start|>", "<|im_end|>", "<|pad|>"):
        raw = raw.replace(tok, "")
    return raw.strip(), new_tokens, elapsed


# ============================================================
# CLI / device helpers
# ============================================================
def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device(value)


def resolve_dtype(value: str, device: torch.device) -> torch.dtype:
    if value == "auto":
        if device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    mapping = {
        "float32": torch.float32, "fp32": torch.float32,
        "float16": torch.float16, "fp16": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    }
    return mapping[value]


def parse_args():
    parser = argparse.ArgumentParser(description="ETET TUI Chat for MiniCPM5 / ETET models.")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--sample", action="store_true", help="Enable sampling instead of greedy.")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="auto",
                        choices=["auto", "float32", "fp32", "float16", "fp16", "bfloat16", "bf16"])
    return parser.parse_args()


# ============================================================
# TUI: chat input (Enter = newline, Ctrl+Enter = send)
# ============================================================
class ChatInput(TextArea):
    BINDINGS = [
        Binding("ctrl+enter", "send_message", "Send", priority=True),
        Binding("alt+enter", "send_message", "Send", priority=True),
    ]

    async def action_send_message(self) -> None:
        app = self.app
        if isinstance(app, ETETChatApp):
            await app.send_current_message()


# ============================================================
# TUI: model selection modal
# ============================================================
class ModelSelectScreen(ModalScreen[Optional[Path]]):
    CSS = """
    ModelSelectScreen {
        align: center middle;
    }
    #model-dialog {
        width: 64;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: thick $primary;
        padding: 1;
    }
    #model-title {
        text-align: center;
        text-style: bold;
        margin: 0 0 1 0;
    }
    #model-list {
        height: auto;
        max-height: 20;
        overflow-y: auto;
    }
    .model-btn {
        width: 100%;
        margin: 0 0 1 0;
    }
    #model-cancel {
        margin: 1 0 0 0;
    }
    """

    def __init__(self, models: List[Path]):
        super().__init__()
        self.models = models

    def compose(self) -> ComposeResult:
        with Vertical(id="model-dialog"):
            yield Static("Select Model", id="model-title")
            with VerticalScroll(id="model-list"):
                for i, model in enumerate(self.models):
                    if is_multimodal_model(model):
                        tag = "ETET-Multimodal"
                    elif is_etet_model(model):
                        tag = "ETET-MoE"
                    else:
                        tag = "Llama"
                    yield Button(f"{model.name}  [{tag}]", id=f"model-{i}", classes="model-btn")
            yield Button("Cancel", id="model-cancel", variant="error")

    @on(Button.Pressed)
    def on_button(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""
        if btn_id == "model-cancel":
            self.dismiss(None)
        elif btn_id.startswith("model-"):
            try:
                idx = int(btn_id.split("-")[-1])
                self.dismiss(self.models[idx])
            except (ValueError, IndexError):
                self.dismiss(None)


# ============================================================
# TUI: image path input modal
# ============================================================
class ImagePathScreen(ModalScreen[Optional[str]]):
    CSS = """
    ImagePathScreen {
        align: center middle;
    }
    #image-dialog {
        width: 64;
        height: auto;
        background: $surface;
        border: thick $primary;
        padding: 1;
    }
    #image-input {
        margin: 1 0;
    }
    #image-buttons {
        height: auto;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="image-dialog"):
            yield Static("Enter image file path:", id="image-title")
            yield Input(placeholder="/path/to/image.png", id="image-input")
            with Horizontal(id="image-buttons"):
                yield Button("OK", id="image-ok", variant="primary")
                yield Button("Cancel", id="image-cancel", variant="error")

    @on(Input.Submitted, "#image-input")
    def submit(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip())

    @on(Button.Pressed)
    def on_button(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""
        if btn_id == "image-ok":
            self.dismiss(self.query_one("#image-input", Input).value.strip())
        elif btn_id == "image-cancel":
            self.dismiss(None)


# ============================================================
# TUI: main chat application
# ============================================================
class ETETChatApp(App):
    TITLE = "ETET Chat"

    CSS = """
    #messages {
        height: 1fr;
        padding: 0 1;
    }
    .message {
        margin: 0 0 1 0;
        padding: 0 1;
        border-left: solid #444444;
    }
    .message.user {
        border-left: solid #4a9e4a;
        background: #1e241e;
    }
    .message.assistant {
        border-left: solid #9a74c4;
        background: #1e1a24;
    }
    .message.system {
        border-left: solid #666666;
        color: #999999;
    }
    .message.error {
        border-left: solid #cc4444;
        color: #ee8888;
        background: #241a1a;
    }
    .debug-line {
        color: #666666;
        margin: 0 0 1 0;
        padding: 0 1;
    }
    #input-area {
        height: auto;
        padding: 0 1;
        border-top: solid #444444;
    }
    #image-info {
        height: auto;
        padding: 0 1;
        color: #4a9e4a;
    }
    #input-field {
        height: 6;
    }
    #button-bar {
        height: auto;
        padding: 1 0;
    }
    #button-bar Button {
        margin: 0 1;
        min-width: 10;
    }
    """

    BINDINGS = [
        Binding("ctrl+m", "select_model", "Model"),
        Binding("ctrl+l", "clear_chat", "Clear"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.max_new_tokens = args.max_new_tokens
        self.do_sample = args.sample
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.device = resolve_device(args.device)
        self.dtype = resolve_dtype(args.dtype, self.device)
        self.model = None
        self.tokenizer = None
        self.mm_components = None
        self.model_dir: Optional[Path] = None
        self.conversation: List[Dict[str, str]] = []
        self.debug_log: List[str] = []
        self.current_image: Optional[PILImage.Image] = None
        self.current_image_path: Optional[str] = None
        self.busy = False
        self.is_loading = False
        self.gen_count = 0
        self.total_tokens = 0
        self._generating_widget: Optional[Static] = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with VerticalScroll(id="messages"):
            yield Static(
                "Welcome to ETET Chat.\n"
                "Ctrl+M: select model | Ctrl+Enter: send | Enter: newline\n"
                "Buttons: Model / Image / Clear / Copy / Exit",
                classes="message system",
            )
        with Vertical(id="input-area"):
            yield Static("", id="image-info")
            yield ChatInput(id="input-field", show_line_numbers=False, soft_wrap=True)
            with Horizontal(id="button-bar"):
                yield Button("Model", id="btn-model", variant="primary")
                yield Button("Image", id="btn-image")
                yield Button("Clear", id="btn-clear", variant="warning")
                yield Button("Copy", id="btn-copy", variant="success")
                yield Button("Exit", id="btn-exit", variant="error")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#input-field", ChatInput).focus()
        self.set_status("Select a model to begin (Ctrl+M)")
        self.call_after_refresh(self.action_select_model)

    # --------------------------------------------------------
    # UI helpers
    # --------------------------------------------------------
    def set_status(self, text: str) -> None:
        self.sub_title = text

    def log_debug(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.debug_log.append(f"[{timestamp}] {message}")
        self.mount_message(f"[dim]{escape(message)}[/dim]", classes="debug-line")

    def log_debug_only(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.debug_log.append(f"[{timestamp}] {message}")

    def mount_message(self, content: str, classes: str = "") -> None:
        messages = self.query_one("#messages")
        widget = Static(content, classes=classes)
        messages.mount(widget)
        messages.scroll_end(animate=False)
        self.call_after_refresh(messages.scroll_end, animate=False)

    def add_conversation_message(self, role: str, content: str) -> None:
        self.conversation.append({"role": role, "content": content})
        if role == "user":
            label = "[b]You[/b]"
            display = escape(content)
            cls = "message user"
        else:
            label = "[b]ETET[/b]"
            display = self._format_assistant(content)
            cls = "message assistant"
        self.mount_message(f"{label}\n{display}", classes=cls)

    def _format_assistant(self, text: str) -> str:
        # Render the think block in green italic, keep the answer below it.
        think, main = split_think(text)
        parts = []
        if think:
            parts.append(f"[green italic]Thinking:\\n{escape(think)}[/green italic]")
        if main:
            parts.append(escape(main))
        if not parts:
            parts.append(escape(text))
        return "\\n\\n".join(parts)

    def _remove_generating_widget(self) -> None:
        if self._generating_widget is not None:
            try:
                self._generating_widget.remove()
            except Exception:
                pass
            self._generating_widget = None

    # --------------------------------------------------------
    # Model selection and loading
    # --------------------------------------------------------
    def action_select_model(self) -> None:
        models = find_models()
        if not models:
            self.mount_message(
                f"No model directories found in: {MODELS_DIR}", classes="message error"
            )
            return
        self.push_screen(ModelSelectScreen(models), self.on_model_selected)

    def on_model_selected(self, model_dir: Optional[Path]) -> None:
        if model_dir is None:
            return
        if self.is_loading or self.busy:
            self.notify("Busy, please wait...", severity="warning")
            return
        self.is_loading = True
        self.set_status(f"Loading {model_dir.name}...")
        self.load_model_worker(model_dir)

    @work(thread=True, exclusive=True)
    def load_model_worker(self, model_dir: Path) -> None:
        try:
            log = lambda m: self.call_from_thread(self.log_debug, m)  # noqa: E731
            log(f"Loading model from: {model_dir}")
            etet = is_etet_model(model_dir)
            multimodal = is_multimodal_model(model_dir)
            model_type = "ETET-Multimodal" if multimodal else ("ETET-MoE" if etet else "LlamaForCausalLM")
            log(f"Model type: {model_type} | device={self.device} | dtype={self.dtype}")

            if self.model is not None or self.mm_components is not None:
                self.model = None
                self.mm_components = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                log("Released previous model.")

            tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
            language_dir = model_dir / "language_model" if multimodal else model_dir

            if etet:
                config = AutoConfig.from_pretrained(language_dir, local_files_only=True)
                validate_config(config)
                model = LlamaForCausalLM(config)
                hidden_size = config.hidden_size
                for layer_idx in range(MOE_START_LAYER, MOE_END_LAYER):
                    layer = model.model.layers[layer_idx]
                    layer.mlp = ETETMoE(layer.mlp, hidden_size)
                load_safetensors_into_model(model, language_dir, log)
                gc_path = language_dir / "generation_config.json"
                if gc_path.exists():
                    try:
                        model.generation_config = GenerationConfig.from_pretrained(
                            language_dir, local_files_only=True
                        )
                    except Exception:
                        pass
                model.to(device=self.device, dtype=self.dtype)
                model.eval()
            else:
                model = LlamaForCausalLM.from_pretrained(
                    language_dir, torch_dtype=self.dtype,
                    local_files_only=True, low_cpu_mem_usage=True,
                )
                model.to(self.device)
                model.eval()

            mm_components = None
            if multimodal:
                if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
                    log("Resizing token embeddings to match tokenizer.")
                    model.resize_token_embeddings(len(tokenizer))
                mm_components = load_multimodal_components(
                    model_dir, model, tokenizer, self.dtype, self.device, log
                )

            param_count = sum(p.numel() for p in model.parameters())
            self.call_from_thread(
                self._on_model_loaded, model_dir, model, tokenizer,
                mm_components, param_count, etet, multimodal,
            )
        except Exception as exc:
            self.call_from_thread(
                self._on_model_load_failed, model_dir, str(exc), traceback.format_exc()
            )

    def _on_model_loaded(self, model_dir, model, tokenizer, mm_components,
                         param_count, etet, multimodal) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.mm_components = mm_components
        self.model_dir = model_dir
        self.is_loading = False
        model_type = "ETET-Multimodal" if multimodal else ("ETET-MoE" if etet else "LlamaForCausalLM")
        params = format_params(param_count)
        self.mount_message(
            f"Model ready: [b]{escape(model_dir.name)}[/b] "
            f"({model_type}, {params} parameters)",
            classes="message system",
        )
        self.log_debug_only(
            f"Model loaded: {model_dir.name} | {model_type} | {params} params "
            f"| device={self.device} | dtype={self.dtype}"
        )
        self.set_status(f"{model_dir.name} | {model_type} | {params}")
        self.query_one("#input-field", ChatInput).focus()

    def _on_model_load_failed(self, model_dir, error: str, tb: str) -> None:
        self.is_loading = False
        self.mount_message(
            f"Failed to load model {escape(model_dir.name)}: {error}",
            classes="message error",
        )
        self.log_debug_only(f"Model load error: {error}")
        self.log_debug_only(tb)
        self.set_status("No model loaded (Ctrl+M to retry)")

    # --------------------------------------------------------
    # Image handling
    # --------------------------------------------------------
    def action_upload_image(self) -> None:
        if self.model is None:
            self.notify("Load a model first (Ctrl+M)", severity="warning")
            return
        if self.mm_components is None:
            self.notify("Current model is text-only, images are not supported", severity="warning")
            return
        self.push_screen(ImagePathScreen(), self.on_image_path)

    def on_image_path(self, path: Optional[str]) -> None:
        if not path:
            return
        try:
            resolved = Path(path).expanduser()
            image = PILImage.open(resolved)
            image.load()
            self.current_image = image
            self.current_image_path = str(resolved)
            w, h = image.size
            size_kb = resolved.stat().st_size / 1024
            self.query_one("#image-info").update(
                f"Image attached: {resolved.name} ({w}x{h}, {size_kb:.1f} KB) "
                f"- will be sent with next message"
            )
            self.log_debug(f"Image loaded: {resolved} ({w}x{h})")
        except Exception as exc:
            self.mount_message(f"Failed to load image: {exc}", classes="message error")

    # --------------------------------------------------------
    # Clear / Copy / Exit
    # --------------------------------------------------------
    def action_clear_chat(self) -> None:
        self.conversation.clear()
        self.debug_log.clear()
        self.current_image = None
        self.current_image_path = None
        self.gen_count = 0
        self.total_tokens = 0
        self.query_one("#messages").remove_children()
        self.query_one("#image-info").update("")
        self.mount_message("Context cleared.", classes="message system")
        if self.model_dir is not None:
            self.set_status(f"{self.model_dir.name} | ready")
        else:
            self.set_status("Select a model to begin (Ctrl+M)")

    def build_transcript(self) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("ETET Chat Transcript")
        lines.append(f"Model: {self.model_dir.name if self.model_dir else '(none)'}")
        if self.current_image_path:
            lines.append(f"Attached image: {self.current_image_path}")
        lines.append(f"Exported: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 60)
        lines.append("")
        for msg in self.conversation:
            lines.append(f"[{msg['role'].upper()}]")
            lines.append(msg["content"])
            lines.append("")
        if self.debug_log:
            lines.append("=" * 60)
            lines.append("DEBUG LOG")
            lines.append("=" * 60)
            for entry in self.debug_log:
                lines.append(entry)
        return "\n".join(lines)

    def action_copy_chat(self) -> None:
        # Transcript keeps full text including <think>...</think> blocks.
        text = self.build_transcript()
        export_path = PROJECT_ROOT / "chat_export.txt"
        saved = False
        try:
            export_path.write_text(text, encoding="utf-8")
            saved = True
        except Exception:
            saved = False
        try:
            copy_via_osc52(text)
            where = "clipboard" + (f" (backup: {export_path.name})" if saved else "")
        except Exception:
            where = f"file: {export_path}" if saved else "nowhere (failed)"
        self.notify(f"Transcript copied to {where}")

    # --------------------------------------------------------
    # Sending and generation
    # --------------------------------------------------------
    async def send_current_message(self) -> None:
        if self.is_loading:
            self.notify("Model is loading, please wait...", severity="warning")
            return
        if self.busy:
            self.notify("Still generating, please wait...", severity="warning")
            return
        if self.model is None or self.tokenizer is None:
            self.notify("No model loaded. Press Ctrl+M or click [Model].", severity="warning")
            return
        input_widget = self.query_one("#input-field", ChatInput)
        text = input_widget.text.strip()
        if not text:
            return
        input_widget.text = ""
        input_widget.focus()
        self.add_conversation_message("user", text)
        if self.current_image is not None and self.mm_components is None:
            self.log_debug("Image attached but model is text-only; image ignored.")
        messages = [dict(m) for m in self.conversation]
        self.busy = True
        self.set_status("Generating...")
        self._generating_widget = Static(
            "[dim italic]Generating response...[/dim italic]", classes="message system"
        )
        self.query_one("#messages").mount(self._generating_widget)
        self.generate_worker(messages)

    @work(thread=True, exclusive=True)
    def generate_worker(self, messages: List[Dict[str, str]]) -> None:
        try:
            text, new_tokens, elapsed = run_generation(
                model=self.model,
                tokenizer=self.tokenizer,
                messages=messages,
                device=self.device,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_p=self.top_p,
                image=self.current_image,
                mm_components=self.mm_components,
            )
            self.call_from_thread(self._on_generation_done, text, new_tokens, elapsed)
        except Exception as exc:
            self.call_from_thread(
                self._on_generation_failed, str(exc), traceback.format_exc()
            )

    def _on_generation_done(self, text: str, new_tokens: int, elapsed: float) -> None:
        self._remove_generating_widget()
        if text:
            self.add_conversation_message("assistant", text)
        else:
            self.mount_message("(empty response)", classes="message system")
        speed = new_tokens / elapsed if elapsed > 0 else 0.0
        self.gen_count += 1
        self.total_tokens += new_tokens
        self.log_debug(
            f"Generated {new_tokens} tokens in {elapsed:.2f}s ({speed:.2f} tok/s)"
        )
        self.set_status(f"Ready | turns={self.gen_count} tokens={self.total_tokens}")
        self.busy = False
        self.query_one("#input-field", ChatInput).focus()

    def _on_generation_failed(self, error: str, tb: str) -> None:
        self._remove_generating_widget()
        self.mount_message(f"Generation failed: {error}", classes="message error")
        self.log_debug_only(f"Generation error: {error}")
        self.log_debug_only(tb)
        self.set_status("Ready (last generation failed)")
        self.busy = False
        self.query_one("#input-field", ChatInput).focus()

    # --------------------------------------------------------
    # Button handlers (mouse clicks)
    # --------------------------------------------------------
    @on(Button.Pressed, "#btn-model")
    def btn_model(self) -> None:
        self.action_select_model()

    @on(Button.Pressed, "#btn-image")
    def btn_image(self) -> None:
        self.action_upload_image()

    @on(Button.Pressed, "#btn-clear")
    def btn_clear(self) -> None:
        self.action_clear_chat()

    @on(Button.Pressed, "#btn-copy")
    def btn_copy(self) -> None:
        self.action_copy_chat()

    @on(Button.Pressed, "#btn-exit")
    def btn_exit(self) -> None:
        self.exit()


def main() -> int:
    args = parse_args()
    app = ETETChatApp(args)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
