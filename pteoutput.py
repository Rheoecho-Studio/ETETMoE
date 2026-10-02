#!/usr/bin/env python3
"""ETET Step 7: Export the final safetensors multimodal model to ExecuTorch .pte files."""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as safe_load_file

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
PTE_DIR = MODELS_DIR / "pte"
LOG_FILE = OUTPUT_DIR / "pteoutput_output.log"
MODEL_NAME_FILE = PROJECT_ROOT / "modelname.txt"
DEFAULT_MODEL_NAME = "ETET"

NUM_EXPERTS = 3
MOE_START_LAYER = 16
MOE_END_LAYER = 23
NUM_TOTAL_LAYERS = 24
IMAGE_SIZE = 512
IMAGE_TOKEN = "<image>"   # special placeholder token the chat app must inject
DEFAULT_EXCLUDE = "router,lm_head,connector"
TARGETS = ("fp16-xnnpack", "fp16-vulkan", "int8-xnnpack",
           "fp16-xnnpack-mm", "fp16-vulkan-mm", "int8-xnnpack-mm")
TARGET_ORDER = ("fp16-xnnpack", "int8-xnnpack", "fp16-vulkan",
                "fp16-xnnpack-mm", "int8-xnnpack-mm", "fp16-vulkan-mm")
# Per-target defaults: ctx is the KV cache length, max_seq bounds tokens per forward call.
# -mm targets fold vision encoding into a single `forward` method (for runtimes that can
# only call one method). max_seq must be >= num_image_tokens (~1024) + text length.
TARGET_DEFAULTS = {
    "fp16-vulkan": {"ctx": 8192, "max_seq": 768},
    "fp16-xnnpack": {"ctx": 5120, "max_seq": 512},
    "int8-xnnpack": {"ctx": 3072, "max_seq": 256},
    "fp16-xnnpack-mm": {"ctx": 5120, "max_seq": 1536},
    "fp16-vulkan-mm": {"ctx": 8192, "max_seq": 2048},
    "int8-xnnpack-mm": {"ctx": 3072, "max_seq": 1280},
}
# Hard ceiling so a long prefill cannot blow up memory on an 8 GB GPU such as the RTX 4060.
MAX_SEQ_CEILING = 2048


def setup_logging():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("etet_pteoutput")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    # Keep messages out of the root logger, otherwise third party basicConfig calls duplicate them.
    logger.propagate = False

    # Separate logger for interactive menus: file only, so the terminal shows them exactly once.
    ui = logging.getLogger("etet_pteoutput.ui")
    ui.setLevel(logging.INFO)
    ui.handlers.clear()
    ui.addHandler(fh)
    ui.propagate = False
    return logger, ui


LOGGER, UI_LOGGER = setup_logging()


def show(message: str = "") -> None:
    # Print to the terminal once and mirror into the log file without duplicating the console line.
    print(message)
    stripped = message.strip()
    if stripped:
        UI_LOGGER.info(stripped)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    # Targets, ctx and max-seq are chosen interactively; only peripheral options stay on the CLI.
    parser = argparse.ArgumentParser(description="Export ETET multimodal model to ExecuTorch .pte files.")
    parser.add_argument("--model", type=Path, default=None, help="Exported multimodal model directory under models/.")
    parser.add_argument("--quant-exclude", type=str, default=DEFAULT_EXCLUDE,
                        help="Comma separated fqn tags kept in FP16 for the INT8 target.")
    parser.add_argument("--eval", action="store_true", help="Run ViT cosine and LM top-1 agreement evaluation.")
    parser.add_argument("--eval-samples", type=int, default=16, help="Identity prompts used for LM evaluation.")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip the ExecuTorch runtime smoke test.")
    return parser.parse_args()


def load_model_names(model_dir: Path):
    # modelname.txt carries two names, one per line, because the exported file
    # name is not necessarily the same as the model's own name:
    #   line 1 = model      -> recorded as the "model" field in the metadata
    #   line 2 = modelname  -> the exported .pte / .json file name prefix
    # A missing or empty file falls back to the defaults (the model directory
    # name and DEFAULT_MODEL_NAME); any other line count is an error.
    fallback_model = model_dir.name
    if not MODEL_NAME_FILE.exists():
        LOGGER.warning("modelname.txt not found; using defaults (model=%s, name=%s).",
                       fallback_model, DEFAULT_MODEL_NAME)
        return fallback_model, DEFAULT_MODEL_NAME
    lines = [ln.strip() for ln in MODEL_NAME_FILE.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        LOGGER.info("modelname.txt is empty; using defaults (model=%s, name=%s).",
                    fallback_model, DEFAULT_MODEL_NAME)
        return fallback_model, DEFAULT_MODEL_NAME
    if len(lines) != 2:
        raise RuntimeError(
            "modelname.txt must contain exactly two lines: line 1 = model name, "
            f"line 2 = exported model name (modelname); found {len(lines)} line(s): {lines}")
    LOGGER.info("Loaded from %s: model=%s, modelname=%s", MODEL_NAME_FILE.name, lines[0], lines[1])
    return lines[0], lines[1]


def choose_model() -> Path:
    candidates = sorted(p for p in MODELS_DIR.iterdir()
                        if p.is_dir() and (p / "etet_multimodal_metadata.json").exists())
    if not candidates:
        raise FileNotFoundError("No exported multimodal model under models/ (missing etet_multimodal_metadata.json).")
    show("\n" + "=" * 72)
    show("Select the model to export")
    show("=" * 72)
    for i, p in enumerate(candidates, 1):
        show(f"{i}. {p.name}")
    while True:
        raw = input(f"Enter choice [1-{len(candidates)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(candidates):
            return candidates[int(raw) - 1]
        show("Invalid choice, try again.")


def choose_targets() -> list:
    # Interactive multi-select: one target, any two, or all three.
    show("\n" + "=" * 72)
    show("Select the PTE targets to package")
    show("=" * 72)
    for i, target in enumerate(TARGETS, 1):
        defaults = TARGET_DEFAULTS[target]
        show(f"{i}. {target:<14s} (ctx={defaults['ctx']}, max-seq={defaults['max_seq']})")
    show(f"{len(TARGETS) + 1}. all")
    show("You may enter several numbers separated by commas, for example: 1,3")
    mapping = {str(i): target for i, target in enumerate(TARGETS, 1)}
    mapping[str(len(TARGETS) + 1)] = "all"
    while True:
        raw = input(f"Enter choice (e.g. 1 / 1,3 / {len(TARGETS) + 1}): ").strip()
        if raw in mapping and mapping[raw] == "all":
            return list(TARGET_ORDER)
        picked, invalid = [], False
        for part in raw.replace(" ", "").split(","):
            if part in mapping and mapping[part] != "all":
                if mapping[part] not in picked:
                    picked.append(mapping[part])
            else:
                invalid = True
        if not invalid and picked:
            return picked
        show("Invalid choice, try again.")


def validate_options(target: str, ctx: int, max_seq: int) -> str:
    if ctx < 1:
        return "ctx must be >= 1."
    if max_seq < 1:
        return "max-seq must be >= 1."
    if max_seq > ctx:
        return f"max-seq ({max_seq}) must not exceed ctx ({ctx})."
    if max_seq > MAX_SEQ_CEILING:
        return f"max-seq ({max_seq}) exceeds the ceiling {MAX_SEQ_CEILING} (8 GB GPU budget)."
    return ""


def prompt_target_options(target: str) -> dict:
    # Ask per target whether to keep the defaults, and re-prompt until values are valid.
    defaults = TARGET_DEFAULTS[target]
    show(f"\n{target}: default ctx={defaults['ctx']}, max-seq={defaults['max_seq']}")
    answer = input("Use the default configuration? [Y/n]: ").strip().lower()
    if answer not in ("n", "no"):
        return dict(defaults)

    ctx, max_seq = defaults["ctx"], defaults["max_seq"]
    while True:
        raw_ctx = input(f"  ctx [{defaults['ctx']}]: ").strip()
        raw_seq = input(f"  max-seq [{defaults['max_seq']}]: ").strip()
        ctx = int(raw_ctx) if raw_ctx.lstrip("-").isdigit() else defaults["ctx"]
        max_seq = int(raw_seq) if raw_seq.lstrip("-").isdigit() else defaults["max_seq"]
        error = validate_options(target, ctx, max_seq)
        if not error:
            return {"ctx": ctx, "max_seq": max_seq}
        show(f"  Invalid: {error} Please try again.")


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def find_model_weight_files(model_dir: Path):
    return sorted(p for p in model_dir.rglob("*") if p.is_file() and p.suffix == ".safetensors")


def load_state_dict_from_dir(model_dir: Path):
    files = find_model_weight_files(model_dir)
    if not files:
        raise FileNotFoundError(f"No safetensors weight files under {model_dir}")
    state_dict = {}
    for p in files:
        state_dict.update(safe_load_file(str(p), device="cpu"))
    LOGGER.info("Loaded %d tensors (%.2f GB) from %s.", len(state_dict),
                sum(t.numel() * t.element_size() for t in state_dict.values()) / 1024 ** 3, model_dir)
    return state_dict


def find_vision_dir(model_dir: Path) -> Path:
    vision_dir = model_dir / "vision_tower"
    if not vision_dir.exists():
        raise FileNotFoundError(f"vision_tower directory not found under {model_dir}")
    return vision_dir


# ---------------------------------------------------------------------------
# Export friendly modules
# ---------------------------------------------------------------------------

class ExportETETMoE(nn.Module):
    # Dense MoE for export: computes all experts and combines with one-hot router weights.
    def __init__(self, dense_mlp: nn.Module, hidden_size: int, num_experts: int = NUM_EXPERTS):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([deepcopy(dense_mlp) for _ in range(num_experts)])
        self.router = nn.ModuleDict({"linear": nn.Linear(hidden_size, num_experts, bias=False)})

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits = self.router["linear"](hidden_states)
        router_probs = torch.softmax(router_logits.float(), dim=-1).to(hidden_states.dtype)
        top_index = torch.argmax(router_probs, dim=-1)
        onehot = F.one_hot(top_index, num_classes=self.num_experts).to(hidden_states.dtype)
        output = torch.zeros_like(hidden_states)
        for expert_idx, expert in enumerate(self.experts):
            gate = onehot[..., expert_idx:expert_idx + 1]
            output = output + gate * expert(hidden_states)
        return output


class ETETVisionConnector(nn.Module):
    # Must match visiontrain.ETETVisionConnector so saved safetensors keys line up.
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


class VisionEncoder(nn.Module):
    # SigLIP-HD vision tower followed by the connector. Static 512x512 input.
    def __init__(self, vision_model: nn.Module, connector: nn.Module):
        super().__init__()
        self.vision = vision_model
        self.connector = connector

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        output = self.vision(pixel_values=pixel_values)
        return self.connector(output.last_hidden_state)


def resolve_rope_theta(config) -> float:
    theta = getattr(config, "rope_theta", None)
    if theta is not None:
        return float(theta)
    rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(rope_parameters, dict) and "rope_theta" in rope_parameters:
        return float(rope_parameters["rope_theta"])
    LOGGER.warning("rope_theta not found in config; falling back to 10000.0.")
    return 10000.0


class TextDecoder(nn.Module):
    # Manual decoder loop with a static KV cache passed via IO, export friendly.
    def __init__(self, language_model: nn.Module, ctx: int):
        super().__init__()
        inner = language_model.model
        self.embed_tokens = inner.embed_tokens
        self.layers = inner.layers
        self.norm = inner.norm
        self.lm_head = language_model.lm_head

        config = language_model.config
        self.ctx = ctx
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(getattr(config, "num_key_value_heads", 0) or config.num_attention_heads)
        self.head_dim = int(getattr(config, "head_dim", 0) or config.hidden_size // self.num_heads)
        self.hidden_size = int(config.hidden_size)
        self.n_rep = self.num_heads // self.num_kv_heads
        self.vocab_size = int(config.vocab_size)

        theta = resolve_rope_theta(config)
        inv_freq = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        positions = torch.arange(ctx, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("rope_cos", emb.cos().to(torch.float16), persistent=False)
        self.register_buffer("rope_sin", emb.sin().to(torch.float16), persistent=False)

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        # q: [1, nh, S, hd], cos/sin: [S, hd]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        def rotate_half(t: torch.Tensor) -> torch.Tensor:
            half = t.shape[-1] // 2
            x1 = t[..., :half]
            x2 = t[..., half:]
            return torch.cat((-x2, x1), dim=-1)

        return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin

    def make_cache(self, device=None, dtype=None) -> torch.Tensor:
        # The cache must match the compute dtype, otherwise SDPA sees mixed q/k/v dtypes.
        device = device if device is not None else self.rope_cos.device
        dtype = dtype if dtype is not None else self.rope_cos.dtype
        return torch.zeros(NUM_TOTAL_LAYERS, 2, self.ctx, self.num_kv_heads, self.head_dim,
                           device=device, dtype=dtype)

    def forward(self, tokens: torch.Tensor, cache_in: torch.Tensor, pos: torch.Tensor):
        x = self.embed_tokens(tokens)
        return self._decode(x, cache_in, pos)

    def _decode(self, x: torch.Tensor, cache_in: torch.Tensor, pos: torch.Tensor):
        seq_len = x.shape[1]
        device = x.device
        cache_in = cache_in.to(x.dtype)

        idx = pos.reshape(1) + torch.arange(seq_len, device=device)
        cos = self.rope_cos.index_select(0, idx).to(x.dtype)
        sin = self.rope_sin.index_select(0, idx).to(x.dtype)

        all_positions = torch.arange(self.ctx, device=device).view(1, self.ctx)
        valid = all_positions <= idx.view(-1, 1)
        neg = torch.finfo(x.dtype).min
        mask = (~valid).to(x.dtype) * neg

        new_caches = []
        for layer_idx, layer in enumerate(self.layers):
            h = layer.input_layernorm(x)
            q = layer.self_attn.q_proj(h).view(1, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = layer.self_attn.k_proj(h).view(1, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = layer.self_attn.v_proj(h).view(1, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            q, k = self._apply_rope(q, k, cos, sin)

            k_seq = k.transpose(1, 2).reshape(seq_len, self.num_kv_heads, self.head_dim)
            v_seq = v.transpose(1, 2).reshape(seq_len, self.num_kv_heads, self.head_dim)
            new_kv = torch.stack((k_seq, v_seq), dim=0)
            cache_l = cache_in[layer_idx].index_copy(1, idx, new_kv)
            new_caches.append(cache_l)

            k_full = cache_l[0].transpose(0, 1).unsqueeze(0)
            v_full = cache_l[1].transpose(0, 1).unsqueeze(0)
            k_full = k_full.repeat_interleave(self.n_rep, dim=1)
            v_full = v_full.repeat_interleave(self.n_rep, dim=1)

            attn = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=mask)
            attn = attn.transpose(1, 2).reshape(1, seq_len, self.num_heads * self.head_dim)
            x = x + layer.self_attn.o_proj(attn)

            h2 = layer.post_attention_layernorm(x)
            x = x + layer.mlp(h2)

        x = self.norm(x)
        logits = self.lm_head(x[:, -1, :])
        cache_out = torch.stack(new_caches, dim=0)
        return logits, cache_out


class TextDecoderMM(TextDecoder):
    # Single-method multimodal decoder: folds vision encoding into `forward` so the
    # model runs under hosts that can only invoke one entry point (a single
    # `forward()` signature). The image placeholder tokens (<image>)
    # MUST occupy positions 1..num_image_tokens (immediately after BOS). The encoder
    # always executes; its output is injected only when `has_image`==1 (multiplied,
    # so there is no data-dependent control flow for torch.export to trace).
    def __init__(self, text_decoder: "TextDecoder", encoder: nn.Module,
                 image_token_id: int, num_image_tokens: int):
        # Reuse the already-built text decoder's weights/buffers.
        self.__dict__.update(text_decoder.__dict__)
        self.encoder = encoder
        self.image_token_id = image_token_id
        self.num_image_tokens = num_image_tokens

    def forward(self, tokens: torch.Tensor, pixel_values: torch.Tensor,
                has_image: torch.Tensor, cache_in: torch.Tensor, pos: torch.Tensor):
        x = self.embed_tokens(tokens)
        img_feats = self.encoder(pixel_values)                 # [1, num_image_tokens, H]
        gate = has_image.reshape(1, 1, 1).to(x.dtype)         # 0 or 1, no branch
        span = self.num_image_tokens
        head = x[:, :1]
        body = x[:, 1:1 + span]
        tail = x[:, 1 + span:]
        injected = (1.0 - gate) * body + gate * img_feats
        x = torch.cat([head, injected, tail], dim=1)
        return self._decode(x, cache_in, pos)


def load_text_model(model_dir: Path, dtype: torch.dtype):
    from transformers import AutoConfig, LlamaForCausalLM
    LOGGER.info("Loading ETET text model from: %s", model_dir)
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    if getattr(config, "num_hidden_layers", None) != NUM_TOTAL_LAYERS:
        raise RuntimeError(f"Expected {NUM_TOTAL_LAYERS} layers, got {config.num_hidden_layers}")

    state_dict = load_state_dict_from_dir(model_dir)
    moe_layers = sorted({int(k.split(".")[2]) for k in state_dict
                         if k.startswith("model.layers.") and ".mlp.experts." in k})
    expected = list(range(MOE_START_LAYER, MOE_END_LAYER + 1))
    if moe_layers != expected:
        raise RuntimeError(f"MoE layers {moe_layers} != expected {expected}")

    model = LlamaForCausalLM(config)
    hidden_size = config.hidden_size
    for layer_idx in expected:
        layer = model.model.layers[layer_idx]
        layer.mlp = ExportETETMoE(layer.mlp, hidden_size, NUM_EXPERTS)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    tag = lambda ks: [k for k in ks if any(k.startswith(f"model.layers.{l}.mlp") for l in expected)]
    if tag(missing):
        raise RuntimeError(f"Missing ETET MoE parameters: {tag(missing)[:10]}")
    if tag(unexpected):
        raise RuntimeError(f"Unexpected ETET MoE parameters: {tag(unexpected)[:10]}")

    model = model.to(dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    LOGGER.info("ETET text model loaded with %d MoE layers.", len(expected))
    return model


def load_vision_encoder(model_dir: Path, dtype: torch.dtype, lm_hidden: int) -> VisionEncoder:
    from transformers import AutoConfig, SiglipVisionModel
    vision_dir = find_vision_dir(model_dir)
    LOGGER.info("Loading SigLIP-HD vision tower from: %s", vision_dir)
    config = AutoConfig.from_pretrained(vision_dir, local_files_only=True)
    if "siglip" not in str(getattr(config, "model_type", "")).lower():
        raise RuntimeError(f"Expected a SigLIP vision tower, got model_type={config.model_type}")
    tower = SiglipVisionModel.from_pretrained(vision_dir, local_files_only=True)
    connector_state = safe_load_file(str(model_dir / "vision_connector" / "connector.safetensors"), device="cpu")
    vision_hidden = int(tower.config.hidden_size)
    connector = ETETVisionConnector(vision_hidden, lm_hidden)
    connector.load_state_dict(connector_state, strict=True)
    LOGGER.info("Connector: %d -> %d", vision_hidden, lm_hidden)
    encoder = VisionEncoder(tower.vision_model, connector)
    return encoder.to(dtype).eval()


# ---------------------------------------------------------------------------
# Export pipeline
# ---------------------------------------------------------------------------

def _fold_quantized_weight(converted, ep, spec, placeholder) -> bool:
    # Fold quantize_per_channel(param) -> dequantize into an int8 constant so the
    # XNNPACK GEMM config sees a static weight input (it requires dequant's input
    # to be a param/constant, not a quantize op). Per-channel scales are computed
    # here as well because the PT2E observers on an exported graph may yield
    # per-tensor scales, which XNNPACK rejects for dynamic quantization.
    qpc = torch.ops.quantized_decomposed.quantize_per_channel.default
    users = list(placeholder.users)
    if len(users) != 1:
        return False
    qnode = users[0]
    if qnode.op != "call_function" or qnode.target != qpc:
        return False
    dq_users = list(qnode.users)
    if len(dq_users) != 1:
        return False
    dequant = dq_users[0]
    axis, qmin, qmax = qnode.args[3], qnode.args[4], qnode.args[5]
    weight = dict(ep.state_dict)[spec.target].detach()
    with torch.no_grad():
        reduce_dims = [d for d in range(weight.dim()) if d != axis]
        amax = weight.abs().amax(dim=reduce_dims, keepdim=True).clamp_min(1e-8)
        scale = (amax / float(-qmin)).reshape(-1)
        zp = torch.zeros_like(scale)
        q = torch.ops.quantized_decomposed.quantize_per_channel.default(
            weight, scale, zp, axis, qmin, qmax, torch.int8)
    graph = converted.graph
    names = {}
    for i, tensor in enumerate((q, scale, zp)):
        name = f"_qw{spec.arg.name}_{i}".replace(".", "_")
        setattr(converted, name, tensor)
        names[i] = name
    with graph.inserting_before(dequant):
        nodes = [graph.create_node("get_attr", names[i]) for i in range(3)]
    dequant.args = (nodes[0], nodes[1], nodes[2]) + tuple(dequant.args[3:])
    old_scale, old_zp = qnode.args[1], qnode.args[2]
    graph.erase_node(qnode)
    # Drop now-unused scale/zero-point get_attr nodes and module buffers.
    for old in (old_scale, old_zp):
        if old is not None and old.op == "get_attr" and not list(old.users):
            graph.erase_node(old)
            if hasattr(converted, old.target):
                delattr(converted, old.target)
    if not list(placeholder.users):
        graph.erase_node(placeholder)
    return True


def pt2e_quantize_and_reexport(ep, quantizer, example_args, source_module,
                               dynamic_shapes=None):
    # Bridge for torch builds without ExportedProgram.from_graph_module: run PT2E
    # quantization on the exported graph module, fold quantized weights into int8
    # constants, then re-export to obtain a fresh ExportedProgram for to_edge.
    # source_module provides the parameter/buffer values (including non-persistent
    # buffers, which are missing from the exported state dict).
    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
    from torch.export.graph_signature import InputKind

    prepared = prepare_pt2e(ep.graph_module, quantizer)
    converted = convert_pt2e(prepared)
    graph = converted.graph
    state = dict(ep.state_dict)
    source_state = {**dict(source_module.named_parameters()),
                    **dict(source_module.named_buffers())}
    user_names = {s.arg.name for s in ep.graph_signature.input_specs
                  if s.kind == InputKind.USER_INPUT}
    for spec in ep.graph_signature.input_specs:
        if spec.kind not in (InputKind.PARAMETER, InputKind.BUFFER):
            continue
        node = next((n for n in list(graph.nodes)
                     if n.op == "placeholder"
                     and (n.target == spec.arg.name or n.name == spec.arg.name
                          or n.target == spec.target)), None)
        if node is None:
            continue
        if spec.kind == InputKind.PARAMETER and _fold_quantized_weight(converted, ep, spec, node):
            continue
        # Re-bind parameters and buffers as module attributes so the re-export
        # treats them as module state instead of required inputs: replace the
        # placeholder with a get_attr node backed by a registered attribute.
        # Attribute names cannot contain dots, so flatten them.
        attr_name = spec.target.replace(".", "_")
        with graph.inserting_before(node):
            attr = graph.create_node("get_attr", attr_name)
        node.replace_all_uses_with(attr)
        graph.erase_node(node)
        if hasattr(converted, attr_name):
            continue
        value = source_state.get(spec.target)
        if value is None:
            value = state.get(spec.target)
        if value is None:
            raise RuntimeError(f"Cannot resolve module state for '{spec.target}'")
        value = value.detach().clone()
        if spec.kind == InputKind.PARAMETER:
            converted.register_parameter(
                attr_name, torch.nn.Parameter(value, requires_grad=False))
        else:
            converted.register_buffer(attr_name, value)
    converted.recompile()
    # A placeholder binds automatically at re-export iff the module holds the
    # attribute; anything else would shift the positional binding of the user
    # input args, so verify no such placeholder remains.
    leftover = [n.name for n in graph.nodes
                if n.op == "placeholder" and n.name not in user_names
                and not hasattr(converted, n.target)]
    if leftover:
        raise RuntimeError(f"Quantized graph still has unbindable placeholders: {leftover}")
    # Reorder the example args to match the graph's user-input placeholder order.
    arg_values = {s.arg.name: v for s, v in zip(
        [s for s in ep.graph_signature.input_specs if s.kind == InputKind.USER_INPUT],
        example_args)}
    ordered_args = tuple(arg_values[n.name] for n in graph.nodes
                         if n.op == "placeholder" and n.name in arg_values)
    return torch.export.export(converted, ordered_args,
                               dynamic_shapes=dynamic_shapes, strict=False)



def build_partitioners(target: str):
    # Strip the "-mm" multimodal suffix before selecting backends.
    if target.endswith("-mm"):
        target = target[:-3]
    if not target.endswith("xnnpack") and "vulkan" not in target:
        raise ValueError(f"Unknown target: {target}")
    from executorch.exir import to_edge_transform_and_lower  # noqa: F401  (availability check)

    if "vulkan" not in target:
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        return [XnnpackPartitioner()], "xnnpack"

    parts = []
    try:
        from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
        parts.append(VulkanPartitioner())
    except ImportError as exc:
        raise RuntimeError(
            "Vulkan backend is not available in this executorch build. "
            "Build executorch from source with -DEXECUTORCH_BUILD_VULKAN=ON "
            "and make sure the Vulkan SDK provides glslc in PATH."
        ) from exc
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    parts.append(XnnpackPartitioner())
    return parts, "vulkan+xnnpack-fallback"


def estimate_delegate_coverage(edge_manager):
    try:
        programs = edge_manager.exported_program()
        items = programs.items() if isinstance(programs, dict) else [("forward", programs)]
        report = {}
        for name, ep in items:
            nodes = list(ep.graph_module.graph.nodes)
            total = sum(1 for n in nodes if n.op == "call_function")
            delegated = sum(1 for n in nodes
                            if n.op == "call_function" and "delegate" in str(n.target).lower())
            report[name] = {"call_function_nodes": total, "delegated_nodes": delegated}
        return report
    except Exception as exc:
        LOGGER.warning("Delegate coverage estimation failed: %s", exc)
        return None


def module_weight_bytes(*modules) -> int:
    total = 0
    for m in modules:
        for t in list(m.parameters()) + list(m.buffers()):
            total += t.numel() * t.element_size()
    return total


def _load_image_token_id(model_dir: Path) -> int:
    # The <image> special token is what the chat app injects into the prompt; the
    # multimodal single-method model replaces those embeddings with vision features.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    tid = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    if tid is None or int(tid) < 0:
        raise RuntimeError(f"Tokenizer has no '{IMAGE_TOKEN}' special token; cannot build multimodal model.")
    LOGGER.info("Image placeholder token '%s' -> id %s", IMAGE_TOKEN, tid)
    return int(tid)


def export_target(model_dir: Path, target: str, args) -> None:
    started = time.time()
    is_mm = target.endswith("-mm")
    base_target = target[:-3] if is_mm else target
    image_token_id = None
    # fp16 for fp16 targets; fp32 for int8: XNNPACK dynamic-quant (qd8) requires
    # fp32 activations, fp16 activations fail XNNCompiler tensor definition.
    dtype = torch.float16 if base_target.startswith("fp16") else torch.float32
    device = torch.device("cpu")
    precision = "fp16" if base_target.startswith("fp16") else "int8"
    LOGGER.info("=" * 72)
    LOGGER.info("Exporting target: %s (model: %s)%s", target, model_dir.name,
                " [single-method multimodal]" if is_mm else "")
    LOGGER.info("=" * 72)

    language_model = load_text_model(model_dir / "language_model", dtype)
    encoder = load_vision_encoder(model_dir, dtype, lm_hidden=int(language_model.config.hidden_size))
    decoder = TextDecoder(language_model, args.ctx).to(dtype).eval()
    for p in decoder.parameters():
        p.requires_grad_(False)
    del language_model
    gc.collect()

    with torch.inference_mode():
        dummy_pixels = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=dtype)
        num_image_tokens = int(encoder(dummy_pixels).shape[1])
    LOGGER.info("Number of image tokens per image: %d", num_image_tokens)

    if is_mm:
        image_token_id = _load_image_token_id(model_dir)
        decoder = TextDecoderMM(decoder, encoder, image_token_id, num_image_tokens).to(dtype).eval()
        for p in decoder.parameters():
            p.requires_grad_(False)
        LOGGER.info("Multimodal decoder: image token id=%s, injected at positions 1..%d",
                    image_token_id, num_image_tokens)

    eval_artifacts = capture_eval_baselines(model_dir, encoder, decoder, args) if (args.eval and not is_mm) else None
    if args.eval and is_mm:
        LOGGER.warning("Evaluation skipped for multimodal target (combined encode needed).")

    if precision == "int8":
        # ExecuTorch int8 dynamic quantization uses the native XNNPACKQuantizer, which
        # annotates and lowers the exported graph to XNNPACK-delegatable int8 ops during
        # the edge transform. torchao's quantize_ produces choose_qparams_affine /
        # quantize_affine ops that ExecuTorch has no kernels for, so it is not used here.
        from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
            XNNPACKQuantizer, get_symmetric_quantization_config)
        quantizer = XNNPACKQuantizer()
        quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True))
        LOGGER.info("INT8 dynamic quantization enabled (XNNPACKQuantizer, per-channel).")
        # XNNPACKQuantizer applies globally (no per-module exclude in this config);
        # the vision tower is quantized together with the rest and runs fine.
        exclude_tags = []
    else:
        quantizer = None
        exclude_tags = []

    from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
    from torch.export import Dim

    partitioners, backend_label = build_partitioners(target)
    LOGGER.info("Partitioners: %s", [type(p).__name__ for p in partitioners])

    pixel_example = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=dtype)

    if args.max_seq > MAX_SEQ_CEILING:
        LOGGER.warning("max-seq=%d exceeds the ceiling of %d (keeps the [S, ctx] attention mask "
                       "and score matrix within an 8 GB GPU budget); clamping to %d.",
                       args.max_seq, MAX_SEQ_CEILING, MAX_SEQ_CEILING)
        args.max_seq = MAX_SEQ_CEILING
    tokens_example = torch.randint(0, decoder.vocab_size, (1, min(8, args.max_seq)))
    cache_example = decoder.make_cache(device)
    pos_example = torch.zeros(1, dtype=torch.int64)

    # INT8 dynamic-quant graphs carry ops outside the core ATen opset (aten._int_mm
    # and quantized_decomposed.*); XNNPACK consumes them via delegation, so allow
    # them through the edge verifier.
    edge_compile_config = None
    if precision == "int8":
        qd = torch.ops.quantized_decomposed
        exc = [torch.ops.aten._int_mm.default]
        for name in ("choose_qparams", "choose_qparams_per_token", "quantize_per_tensor",
                     "dequantize_per_tensor", "quantize_per_channel", "dequantize_per_channel",
                     "quantize_per_channel_group", "dequantize_per_channel_group"):
            ol = getattr(qd, name, None)
            if ol is not None:
                exc.extend(getattr(ol, o) for o in ol.overloads())
        edge_compile_config = EdgeCompileConfig(_core_aten_ops_exception_list=exc)

    if is_mm:
        has_example = torch.tensor(1, dtype=torch.int64)
        # The sequence must always reserve `num_image_tokens` slots right after BOS;
        # the model replaces them with vision features when has_image==1. The export
        # example therefore embeds those slots so the fixed-span slice traces cleanly.
        slot = min(num_image_tokens, max(1, args.max_seq - 2))
        tokens_example = torch.tensor(
            [[0] + [image_token_id] * slot + [decoder.vocab_size - 1]], dtype=torch.int64)
        fwd_ep = torch.export.export(
            decoder, (tokens_example, pixel_example, has_example, cache_example, pos_example),
            dynamic_shapes={"tokens": {1: Dim.AUTO},
                            "pixel_values": None, "has_image": None,
                            "cache_in": None, "pos": None},
            strict=False)
        LOGGER.info("Multimodal forward exported (single method).")
        if quantizer is not None:
            fwd_ep = pt2e_quantize_and_reexport(
                fwd_ep, quantizer,
                (tokens_example, pixel_example, has_example, cache_example, pos_example),
                decoder,
                {"tokens": {1: Dim.AUTO}, "pixel_values": None, "has_image": None,
                 "cache_in": None, "pos": None})
            LOGGER.info("INT8 PT2E quantization applied and re-exported.")
        edge_manager = to_edge_transform_and_lower({"forward": fwd_ep}, partitioner=partitioners,
                                                    compile_config=edge_compile_config)
    else:
        enc_ep = torch.export.export(encoder, (pixel_example,), strict=False)
        LOGGER.info("Vision encoder exported.")
        dynamic_shapes = {"tokens": {1: Dim("seq", min=1, max=args.max_seq)},
                          "cache_in": None, "pos": None}
        fwd_ep = torch.export.export(decoder, (tokens_example, cache_example, pos_example),
                                     dynamic_shapes=dynamic_shapes, strict=False)
        LOGGER.info("Text decoder exported (ctx=%d, max seq per call=%d).",
                    args.ctx, args.max_seq)
        if quantizer is not None:
            enc_ep = pt2e_quantize_and_reexport(enc_ep, quantizer, (pixel_example,), encoder)
            fwd_ep = pt2e_quantize_and_reexport(
                fwd_ep, quantizer, (tokens_example, cache_example, pos_example), decoder,
                dynamic_shapes)
            LOGGER.info("INT8 PT2E quantization applied and re-exported.")
        edge_manager = to_edge_transform_and_lower(
            {"encode_image": enc_ep, "forward": fwd_ep},
            partitioner=partitioners,
            compile_config=edge_compile_config,
        )
    cache_mb = cache_example.numel() * cache_example.element_size() / 1024 ** 2
    LOGGER.info("KV cache: %.0f MB.", cache_mb)

    et_program = edge_manager.to_executorch()
    LOGGER.info("Lowering to ExecuTorch finished.")

    PTE_DIR.mkdir(parents=True, exist_ok=True)
    pte_path = PTE_DIR / f"{args.model_name}_{target}.pte"
    with open(pte_path, "wb") as f:
        f.write(et_program.buffer)
    file_bytes = pte_path.stat().st_size
    expected_bytes = module_weight_bytes(encoder, decoder)
    ratio = file_bytes / max(1, expected_bytes)
    LOGGER.info("Saved %s (%.2f GB).", pte_path, file_bytes / 1024 ** 3)
    LOGGER.info("Size check: file=%.2f GB, weights=%.2f GB, ratio=%.3f",
                file_bytes / 1024 ** 3, expected_bytes / 1024 ** 3, ratio)
    if ratio > 1.3:
        LOGGER.warning("PTE is much larger than the weight sum; multi-method constants may be duplicated.")

    smoke_result = "skipped"
    eval_result = None
    if "xnnpack" in target and not args.skip_smoke:
        smoke_result = run_smoke_test(pte_path, decoder, args, dtype,
                                      is_mm=is_mm, image_token_id=image_token_id)
    elif args.skip_smoke:
        LOGGER.info("Smoke test skipped by user request.")
    else:
        LOGGER.warning("No local runtime for backend %s; skip smoke test.", backend_label)

    if args.eval:
        if is_mm:
            LOGGER.warning("Evaluation for multimodal single-method requires combined encode; skipped.")
        elif "xnnpack" in target:
            eval_result = run_eval_with_ctx(pte_path, eval_artifacts, args.ctx)
        else:
            LOGGER.warning("Evaluation for Vulkan requires a native runner; skipped.")

    metadata = {
        "model": args.source_model,
        "model_name": args.model_name,
        "target": target,
        "backend": backend_label,
        "precision": precision,
        "dtype": "float16",
        "ctx": args.ctx,
        "image_size": IMAGE_SIZE,
        "num_image_tokens": num_image_tokens,
        "multimodal_single_method": is_mm,
        "image_token_id": image_token_id,
        "moe_layers": list(range(MOE_START_LAYER, MOE_END_LAYER + 1)),
        "quantized": precision == "int8",
        "quant_exclude": exclude_tags,
        "file_bytes": file_bytes,
        "param_bytes": expected_bytes,
        "file_param_ratio": round(ratio, 4),
        "delegate_coverage": estimate_delegate_coverage(edge_manager),
        "smoke_test": smoke_result,
        "eval": eval_result,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path = PTE_DIR / f"{args.model_name}_{target}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    LOGGER.info("Metadata written: %s", meta_path)

    copy_tokenizer_files(model_dir)
    LOGGER.info("Target %s finished in %.1f minutes.", target, (time.time() - started) / 60)

    del encoder, decoder, et_program, edge_manager, fwd_ep
    if "enc_ep" in dir():
        del enc_ep
    gc.collect()


def copy_tokenizer_files(model_dir: Path) -> None:
    names = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
    for name in names:
        src = model_dir / name
        if src.exists():
            dst = PTE_DIR / name
            dst.write_bytes(src.read_bytes())
            LOGGER.info("Copied %s -> %s", src.name, dst)


# ---------------------------------------------------------------------------
# Smoke test and evaluation
# ---------------------------------------------------------------------------

def run_smoke_test(pte_path: Path, decoder: TextDecoder, args, dtype: torch.dtype,
                   is_mm: bool = False, image_token_id=None) -> str:
    try:
        from executorch.runtime import Runtime
    except ImportError as exc:
        LOGGER.warning("executorch runtime unavailable, smoke test skipped: %s", exc)
        return "unavailable"
    try:
        runtime = Runtime.get()
        program = runtime.load_program(str(pte_path))

        if is_mm:
            method = program.load_method("forward")
            num_img = decoder.num_image_tokens
            seq = [0] + [image_token_id] * num_img + [decoder.vocab_size - 1]
            tokens = torch.tensor([seq], dtype=torch.int64)
            pixel = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=dtype)
            has_image = torch.tensor(1, dtype=torch.int64)
            cache = decoder.make_cache(tokens.device)
            pos = torch.zeros(1, dtype=torch.int64)
            outputs = method.execute([tokens, pixel, has_image, cache, pos])
            pte_logits = outputs[0]
            LOGGER.info("Smoke forward OK, logits shape: %s", tuple(pte_logits.shape))
            with torch.inference_mode():
                eager_logits, _ = decoder(tokens, pixel, has_image, cache, pos)
            pte_top1 = int(pte_logits.reshape(-1).argmax())
            eager_top1 = int(eager_logits.reshape(-1).argmax())
            LOGGER.info("Smoke top-1: pte=%d eager=%d match=%s", pte_top1, eager_top1, pte_top1 == eager_top1)
        else:
            encode_method = program.load_method("encode_image")
            pixel = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=dtype)
            encode_out = encode_method.execute([pixel])[0]
            LOGGER.info("Smoke encode_image OK, output shape: %s", tuple(encode_out.shape))

            forward_method = program.load_method("forward")
            tokens = torch.randint(0, decoder.vocab_size, (1, 8))
            cache = decoder.make_cache(tokens.device)
            pos = torch.zeros(1, dtype=torch.int64)
            outputs = forward_method.execute([tokens, cache, pos])
            pte_logits = outputs[0]
            LOGGER.info("Smoke forward OK, logits shape: %s", tuple(pte_logits.shape))

            with torch.inference_mode():
                eager_logits, _ = decoder(tokens, cache, pos)
            pte_top1 = int(pte_logits.reshape(-1).argmax())
            eager_top1 = int(eager_logits.reshape(-1).argmax())
            LOGGER.info("Smoke top-1: pte=%d eager=%d match=%s", pte_top1, eager_top1, pte_top1 == eager_top1)

        LOGGER.info("Smoke test passed.")
        return "passed"
    except Exception as exc:
        LOGGER.error("Smoke test failed: %s", exc)
        return "failed"


def load_eval_pixels(vision_dir: Path, limit: int = 4):
    test_dir = PROJECT_ROOT / "test"
    images = sorted(list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png")))[:limit]
    if not images:
        LOGGER.warning("No test images found under %s; vision evaluation skipped.", test_dir)
        return []
    try:
        from transformers import AutoImageProcessor
        processor = AutoImageProcessor.from_pretrained(vision_dir, local_files_only=True)
        use_processor = True
    except Exception as exc:
        LOGGER.warning("AutoImageProcessor unavailable (%s); using manual preprocessing.", exc)
        processor, use_processor = None, False

    from PIL import Image
    pixels = []
    for path in images:
        image = Image.open(path).convert("RGB")
        if use_processor:
            batch = processor(images=image, return_tensors="pt")
            tensor = batch["pixel_values"]
        else:
            from torchvision import transforms
            pipeline = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            tensor = pipeline(image).unsqueeze(0)
        if tensor.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
            tensor = F.interpolate(tensor, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        pixels.append(tensor.to(torch.float16))
    LOGGER.info("Prepared %d evaluation images.", len(pixels))
    return pixels


def build_lm_eval_inputs(model_dir: Path, tokenizer, n_samples: int, ctx: int):
    prompts = []
    try:
        from etet_id_datasets import ETET_ID_DATASETS
        records = [r["conversations"][0]["content"] for r in ETET_ID_DATASETS]
        LOGGER.info("Loaded %d identity prompts from etet_id_datasets.", len(records))
    except Exception as exc:
        LOGGER.warning("etet_id_datasets unavailable (%s); using builtin prompts.", exc)
        records = ["你是谁？", "你是誰？", "Who are you?", "你是谁开发的？"]

    step = max(1, len(records) // max(1, n_samples))
    selected = records[::step][:n_samples]
    for text in selected:
        messages = [{"role": "user", "content": text}]
        try:
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ids = tokenizer(rendered, return_tensors="pt").input_ids
        except Exception:
            ids = tokenizer(text, return_tensors="pt").input_ids
        if ids.shape[1] > ctx - 1:
            ids = ids[:, -(ctx - 1):]
        prompts.append(ids)
    LOGGER.info("Prepared %d LM evaluation prompts.", len(prompts))
    return prompts


def capture_eval_baselines(model_dir: Path, encoder: nn.Module, decoder: TextDecoder, args):
    from transformers import AutoTokenizer
    vision_dir = find_vision_dir(model_dir)
    pixels = load_eval_pixels(vision_dir)
    ref_embeds = []
    with torch.inference_mode():
        for tensor in pixels:
            ref_embeds.append(encoder(tensor).squeeze(0).float().cpu())

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    prompts = build_lm_eval_inputs(model_dir, tokenizer, args.eval_samples, args.ctx)
    ref_top1, ref_logits = [], []
    with torch.inference_mode():
        cache = decoder.make_cache(torch.device("cpu"))
        for ids in prompts:
            logits, _ = decoder(ids, cache, torch.zeros(1, dtype=torch.int64))
            ref_logits.append(logits.squeeze(0).float().cpu())
            ref_top1.append(int(logits.reshape(-1).argmax()))
    LOGGER.info("Captured eager FP16 baselines: %d images, %d prompts.", len(ref_embeds), len(ref_top1))
    return {"pixels": pixels, "ref_embeds": ref_embeds, "prompts": prompts,
            "ref_top1": ref_top1, "ref_logits": ref_logits}


def run_eval_with_ctx(pte_path: Path, artifacts, ctx: int) -> dict:
    from executorch.runtime import Runtime
    runtime = Runtime.get()
    program = runtime.load_program(str(pte_path))
    encode_method = program.load_method("encode_image")
    forward_method = program.load_method("forward")

    cosines = []
    for tensor, ref in zip(artifacts["pixels"], artifacts["ref_embeds"]):
        out = encode_method.execute([tensor])[0].squeeze(0).float().cpu()
        cosines.append(F.cosine_similarity(out, ref, dim=-1).mean().item())
    vision_cosine = sum(cosines) / max(1, len(cosines))
    LOGGER.info("Vision embed cosine: %.5f", vision_cosine)

    cache = torch.zeros(NUM_TOTAL_LAYERS, 2, ctx, 2, 128, dtype=torch.float16)
    agree = 0
    for ids, ref_top1 in zip(artifacts["prompts"], artifacts["ref_top1"]):
        outputs = forward_method.execute([ids, cache, torch.zeros(1, dtype=torch.int64)])
        pte_top1 = int(outputs[0].reshape(-1).argmax())
        agree += int(pte_top1 == ref_top1)
    total = max(1, len(artifacts["prompts"]))
    agreement = agree / total
    LOGGER.info("LM top-1 agreement vs eager FP16: %.3f (%d/%d)", agreement, agree, total)
    return {"vision_cosine": round(vision_cosine, 5),
            "lm_top1_agreement": round(agreement, 4),
            "lm_samples": total}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    torch.set_num_threads(min(16, __import__("os").cpu_count() or 8))
    LOGGER.info("=" * 72)
    LOGGER.info("ETET Step 7: PTE Export")
    LOGGER.info("Python: %s | PyTorch: %s", sys.version.split()[0], torch.__version__)
    LOGGER.info("=" * 72)

    model_dir = args.model if args.model is not None else choose_model()
    if not model_dir.is_absolute():
        model_dir = PROJECT_ROOT / model_dir
    if not (model_dir / "etet_multimodal_metadata.json").exists():
        raise FileNotFoundError(f"Not an exported multimodal model directory: {model_dir}")
    args.source_model, args.model_name = load_model_names(model_dir)
    targets = choose_targets()

    LOGGER.info("Model: %s", model_dir)
    LOGGER.info("Selected targets: %s", targets)

    failures = []
    for t in targets:
        chosen = prompt_target_options(t)
        ctx, max_seq = chosen["ctx"], chosen["max_seq"]
        error = validate_options(t, ctx, max_seq)
        if error:
            LOGGER.error("Target %s skipped: %s", t, error)
            failures.append(t)
            continue
        options = argparse.Namespace(**vars(args))
        options.ctx = ctx
        options.max_seq = max_seq
        LOGGER.info("Target %s resolved: ctx=%d, max-seq=%d", t, ctx, max_seq)
        try:
            export_target(model_dir, t, options)
        except Exception as exc:
            LOGGER.exception("Target %s failed: %s", t, exc)
            failures.append(t)

    LOGGER.info("=" * 72)
    if failures:
        LOGGER.error("Finished with failures: %s", failures)
        return 1
    LOGGER.info("All targets exported successfully. Output directory: %s", PTE_DIR)
    LOGGER.info("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
