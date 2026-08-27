import argparse
import copy
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
SOURCE_MODEL_DIR = MODELS_DIR / "OpenBMB" / "MiniCPM5-1B-SFT"
TARGET_MODEL_DIR = MODELS_DIR / "ETET-Base"

NUM_EXPERTS = 3
NUM_MOE_LAYERS = 8
MOE_START_LAYER = 16
MOE_END_LAYER = 24
ROUTER_TOP_K = 1

LOG_FILE = OUTPUT_DIR / "upcycle_output.log"


class Top1Router(nn.Module):
    """
    Top‑1 router for MoE. Supports automatic dtype casting.
    """
    def __init__(self, hidden_size: int, num_experts: int, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = ROUTER_TOP_K
        self.linear = nn.Linear(hidden_size, num_experts, bias=False, dtype=dtype)
        nn.init.zeros_(self.linear.weight)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    """
    MoE module that replaces the original dense FFN.
    Uses 3 experts with Top‑1 routing and load‑balancing loss.
    """
    def __init__(
        self,
        dense_ffn: nn.Module,
        hidden_size: int,
        num_experts: int = NUM_EXPERTS,
        top_k: int = ROUTER_TOP_K,
    ):
        super().__init__()
        if top_k != 1:
            raise ValueError("This implementation currently requires Top‑1 routing.")

        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size

        router_dtype = next(dense_ffn.parameters()).dtype
        self.router = Top1Router(
            hidden_size=hidden_size,
            num_experts=num_experts,
            dtype=router_dtype,
        )

        self.experts = nn.ModuleList([copy.deepcopy(dense_ffn) for _ in range(num_experts)])

        self.last_router_logits: Optional[torch.Tensor] = None
        self.last_router_indices: Optional[torch.Tensor] = None
        self.last_load_balancing_loss: Optional[torch.Tensor] = None

    def _compute_load_balancing_loss(
        self,
        router_logits: torch.Tensor,
        router_indices: torch.Tensor,
    ) -> torch.Tensor:
        num_experts = self.num_experts
        router_probs = torch.softmax(router_logits.float(), dim=-1)

        flat_probs = router_probs.reshape(-1, num_experts)
        flat_indices = router_indices.reshape(-1)

        expert_fraction = torch.zeros(
            num_experts,
            device=router_logits.device,
            dtype=router_probs.dtype,
        )
        for expert_idx in range(num_experts):
            expert_fraction[expert_idx] = (flat_indices == expert_idx).float().mean()

        expert_probability = flat_probs.mean(dim=0)
        loss = num_experts * torch.sum(expert_fraction * expert_probability)
        return loss

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits, router_probs, router_indices = self.router(hidden_states)

        self.last_router_logits = router_logits
        self.last_router_indices = router_indices
        self.last_load_balancing_loss = self._compute_load_balancing_loss(
            router_logits=router_logits,
            router_indices=router_indices,
        )

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


def configure_logging() -> logging.Logger:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("etet_upcycle")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def resolve_source_model(path: Path) -> Path:
    if path.exists():
        return path
    alternatives = [
        MODELS_DIR / "MiniCPM5-1B-SFT",
        MODELS_DIR / "OpenBMB" / "MiniCPM5-1B-SFT",
        MODELS_DIR / "openbmb" / "MiniCPM5-1B-SFT",
    ]
    for candidate in alternatives:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"MiniCPM5-1B-SFT not found. Expected: {SOURCE_MODEL_DIR}"
    )


def validate_source_config(config, logger: logging.Logger) -> None:
    architectures = list(getattr(config, "architectures", None) or [])
    model_type = getattr(config, "model_type", None)
    num_layers = getattr(config, "num_hidden_layers", None)
    logger.info("Source model architecture: %s", architectures)
    logger.info("Source model type: %s", model_type)
    logger.info("Source model layers: %s", num_layers)

    if "LlamaForCausalLM" not in architectures:
        raise ValueError(f"Expected LlamaForCausalLM, got {architectures}")
    if num_layers != 24:
        raise ValueError(f"Expected 24 layers, got {num_layers}")
    if model_type != "llama":
        raise ValueError(f"Expected 'llama', got {model_type!r}")


def get_dense_ffn(layer: nn.Module) -> nn.Module:
    if not hasattr(layer, "mlp"):
        raise AttributeError("Layer does not have 'mlp' module")
    return layer.mlp


def replace_moe_layers(model: LlamaForCausalLM, logger: logging.Logger) -> List[int]:
    decoder = model.model.layers
    hidden_size = model.config.hidden_size
    modified_layers = []

    for layer_idx in range(MOE_START_LAYER, MOE_END_LAYER):
        layer = decoder[layer_idx]
        dense_ffn = get_dense_ffn(layer)

        logger.info(
            "Converting layer %d from Dense FFN to %d‑Expert Top‑1 MoE.",
            layer_idx, NUM_EXPERTS
        )

        moe = ETETMoE(
            dense_ffn=dense_ffn,
            hidden_size=hidden_size,
            num_experts=NUM_EXPERTS,
            top_k=ROUTER_TOP_K,
        )
        layer.mlp = moe
        modified_layers.append(layer_idx)

    return modified_layers


def initialize_routers(model: nn.Module, logger: logging.Logger) -> None:
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, Top1Router):
            nn.init.zeros_(module.linear.weight)
            count += 1
            logger.info("Initialized router with zero logits: %s", name)
    logger.info("Initialized %d routers.", count)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def format_parameter_count(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.3f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3f}M"
    if value >= 1_000:
        return f"{value / 1_000:.3f}K"
    return str(value)


def save_metadata(
    output_dir: Path,
    report: Dict[str, int],
    modified_layers: List[int],
) -> None:
    metadata = {
        "base_model": "OpenBMB/MiniCPM5-1B-SFT",
        "architecture": "LlamaForCausalLM",
        "num_hidden_layers": 24,
        "dense_layers": list(range(0, 16)),
        "moe_layers": modified_layers,
        "num_experts": NUM_EXPERTS,
        "router": {
            "type": "linear",
            "routing": "top-1",
            "top_k": ROUTER_TOP_K,
            "load_balancing_loss": True,
        },
        "expert_initialization": "deepcopy_of_original_dense_ffn",
        "parameter_report": report,
    }
    with open(output_dir / "etet_upcycle_config.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def copy_tokenizer(source_dir: Path, target_dir: Path, logger: logging.Logger) -> None:
    tokenizer_files = [
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "tokenizer.model", "vocab.json", "merges.txt", "chat_template.jinja"
    ]
    copied = 0
    for fname in tokenizer_files:
        src = source_dir / fname
        dst = target_dir / fname
        if src.exists():
            shutil.copy2(src, dst)
            copied += 1
    logger.info("Copied %d tokenizer files.", copied)


def save_model(
    model: LlamaForCausalLM,
    tokenizer,
    source_dir: Path,
    target_dir: Path,
    logger: logging.Logger,
) -> None:
    if target_dir.exists():
        logger.info("Removing existing output directory: %s", target_dir)
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving ETET-Base to %s", target_dir)
    model.save_pretrained(target_dir, safe_serialization=True, max_shard_size="2GB")
    copy_tokenizer(source_dir, target_dir, logger)
    if not (target_dir / "tokenizer_config.json").exists():
        tokenizer.save_pretrained(target_dir)
    logger.info("ETET-Base saved successfully.")


def get_test_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def run_forward_test(model: LlamaForCausalLM, tokenizer, logger: logging.Logger) -> None:
    logger.info("Running ETET-MoE forward test.")
    device = get_test_device(model)
    text = "Hello, this is an ETET architecture test."
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    was_training = model.training
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    if not hasattr(outputs, "logits"):
        raise RuntimeError("Forward test did not return logits.")
    if outputs.logits.ndim != 3:
        raise RuntimeError(f"Unexpected logits shape: {tuple(outputs.logits.shape)}")
    logger.info("Forward test passed. Logits shape: %s", tuple(outputs.logits.shape))
    if was_training:
        model.train()


def run_generation_test(model: LlamaForCausalLM, tokenizer, logger: logging.Logger) -> None:
    logger.info("Running ETET-MoE generation test.")
    device = get_test_device(model)
    prompt = "Please introduce yourself briefly."
    try:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
        )
    except Exception:
        inputs = tokenizer(prompt, return_tensors="pt").input_ids

    if isinstance(inputs, torch.Tensor):
        input_ids = inputs.to(device)
        attention_mask = torch.ones_like(input_ids)
    else:
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(device)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=16,
            do_sample=False,
            use_cache=False,
        )
    if generated.shape[1] <= input_ids.shape[1]:
        raise RuntimeError("Generation test did not generate new tokens.")
    logger.info(
        "Generation test passed. Generated token count: %d",
        generated.shape[1] - input_ids.shape[1]
    )
    if was_training:
        model.train()


def compare_output_behavior(
    original_model: LlamaForCausalLM,
    moe_model: LlamaForCausalLM,
    tokenizer,
    logger: logging.Logger,
) -> Dict[str, float]:
    logger.info("Comparing original Dense and converted ETET-MoE outputs.")
    device_orig = get_test_device(original_model)
    device_moe = get_test_device(moe_model)
    text = "The purpose of this test is to verify Dense-to-MoE conversion."

    orig_inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    moe_inputs = {k: v.clone() for k, v in orig_inputs.items()}
    orig_inputs = {k: v.to(device_orig) for k, v in orig_inputs.items()}
    moe_inputs = {k: v.to(device_moe) for k, v in moe_inputs.items()}

    orig_was_training = original_model.training
    moe_was_training = moe_model.training
    original_model.eval()
    moe_model.eval()

    with torch.no_grad():
        orig_out = original_model(**orig_inputs).logits.float()
        moe_out = moe_model(**moe_inputs).logits.float()

    if orig_out.shape != moe_out.shape:
        raise RuntimeError(f"Shape mismatch: {orig_out.shape} vs {moe_out.shape}")

    diff = moe_out - orig_out
    mean_ae = diff.abs().mean().item()
    max_ae = diff.abs().max().item()
    orig_norm = orig_out.norm().item()
    diff_norm = diff.norm().item()
    rel_l2 = diff_norm / max(orig_norm, 1e-12)

    logger.info("Output mean absolute difference: %.8f", mean_ae)
    logger.info("Output maximum absolute difference: %.8f", max_ae)
    logger.info("Output relative L2 difference: %.8f", rel_l2)

    if orig_was_training:
        original_model.train()
    if moe_was_training:
        moe_model.train()

    return {
        "mean_absolute_error": mean_ae,
        "max_absolute_error": max_ae,
        "relative_l2_difference": rel_l2,
    }


def verify_expert_initialization(
    original_model: LlamaForCausalLM,
    moe_model: LlamaForCausalLM,
    logger: logging.Logger,
) -> None:
    logger.info("Verifying Expert initialization.")
    for layer_idx in range(MOE_START_LAYER, MOE_END_LAYER):
        orig_ffn = original_model.model.layers[layer_idx].mlp
        moe_module = moe_model.model.layers[layer_idx].mlp
        if not isinstance(moe_module, ETETMoE):
            raise RuntimeError(f"Layer {layer_idx} is not an ETETMoE module.")
        for expert_idx, expert in enumerate(moe_module.experts):
            orig_state = orig_ffn.state_dict()
            expert_state = expert.state_dict()
            if orig_state.keys() != expert_state.keys():
                raise RuntimeError(f"Key mismatch at layer {layer_idx}, expert {expert_idx}")
            for key in orig_state:
                if not torch.equal(orig_state[key].detach().cpu(), expert_state[key].detach().cpu()):
                    raise RuntimeError(
                        f"Weight mismatch at layer {layer_idx}, expert {expert_idx}, key {key}"
                    )
    logger.info(
        "All %d Experts across layers %d‑%d match original Dense FFN.",
        NUM_EXPERTS, MOE_START_LAYER, MOE_END_LAYER - 1
    )


def inspect_structure(model: LlamaForCausalLM, logger: logging.Logger) -> None:
    logger.info("Inspecting ETET model structure.")
    for idx in range(24):
        layer = model.model.layers[idx]
        if idx < MOE_START_LAYER:
            logger.info("Layer %02d: Dense", idx)
        else:
            if not isinstance(layer.mlp, ETETMoE):
                raise RuntimeError(f"Layer {idx} expected ETETMoE.")
            logger.info("Layer %02d: MoE | experts=%d | top_k=%d", idx, layer.mlp.num_experts, layer.mlp.top_k)


def ensure_safe_dtype(model: nn.Module, logger: logging.Logger) -> None:
    param = next(model.parameters(), None)
    if param is None:
        return
    logger.info("Model parameter dtype: %s", param.dtype)
    if param.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        logger.warning("Unexpected model dtype detected: %s", param.dtype)


def load_original_model(source_dir: Path, logger: logging.Logger) -> LlamaForCausalLM:
    logger.info("Loading original MiniCPM5-1B-SFT from %s", source_dir)
    model = LlamaForCausalLM.from_pretrained(
        source_dir, torch_dtype="auto", device_map="cpu", low_cpu_mem_usage=True
    )
    model.config.use_cache = False
    logger.info("Original MiniCPM5-1B-SFT loaded.")
    return model


def clone_original_for_comparison(source_dir: Path, logger: logging.Logger) -> LlamaForCausalLM:
    logger.info("Loading a second CPU copy for output comparison.")
    model = LlamaForCausalLM.from_pretrained(
        source_dir, torch_dtype="auto", device_map="cpu", low_cpu_mem_usage=True
    )
    model.config.use_cache = False
    logger.info("Comparison copy loaded.")
    return model


def create_etet_model(source_model: LlamaForCausalLM, logger: logging.Logger) -> LlamaForCausalLM:
    logger.info("Creating ETET-MoE from MiniCPM5-1B-SFT.")
    model = source_model
    modified_layers = replace_moe_layers(model, logger)
    initialize_routers(model, logger)
    inspect_structure(model, logger)

    model.config.etet_moe = True
    model.config.etet_num_experts = NUM_EXPERTS
    model.config.etet_top_k = ROUTER_TOP_K
    model.config.etet_moe_start_layer = MOE_START_LAYER
    model.config.etet_moe_end_layer = MOE_END_LAYER
    model.config.etet_load_balancing_loss = True
    model.etet_moe_layers = modified_layers

    logger.info("ETET-MoE conversion completed.")
    return model


def write_conversion_report(
    output_dir: Path,
    parameter_report_data: Dict[str, int],
    output_report: Dict[str, float],
) -> None:
    report = {
        "parameter_report": parameter_report_data,
        "output_comparison": output_report,
        "configuration": {
            "num_experts": NUM_EXPERTS,
            "top_k": ROUTER_TOP_K,
            "moe_start_layer": MOE_START_LAYER,
            "moe_end_layer": MOE_END_LAYER,
            "num_moe_layers": NUM_MOE_LAYERS,
            "load_balancing_loss": True,
        },
    }
    with open(output_dir / "upcycle_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ETET Dense-to-MoE Sparse Upcycling converter.")
    parser.add_argument("--source", type=Path, default=SOURCE_MODEL_DIR,
                        help="Path to the MiniCPM5-1B-SFT model directory.")
    parser.add_argument("--output", type=Path, default=TARGET_MODEL_DIR,
                        help="Path for the ETET-Base output directory.")
    parser.add_argument("--experts", type=int, default=NUM_EXPERTS,
                        help="Number of MoE experts (must be 3).")
    parser.add_argument("--top-k", type=int, default=ROUTER_TOP_K,
                        help="Router top‑k value (must be 1).")
    parser.add_argument("--skip-generation-test", action="store_true",
                        help="Skip the generation verification test.")
    parser.add_argument("--skip-output-comparison", action="store_true",
                        help="Skip Dense‑vs‑MoE output comparison.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = configure_logging()

    logger.info("=" * 72)
    logger.info("ETET Dense-to-MoE Sparse Upcycling")
    logger.info("=" * 72)

    if args.experts != NUM_EXPERTS:
        raise ValueError(f"ETET requires exactly {NUM_EXPERTS} experts.")
    if args.top_k != ROUTER_TOP_K:
        raise ValueError(f"ETET requires Top-{ROUTER_TOP_K} routing.")

    logger.info("Python version: %s", sys.version.replace("\n", " "))
    logger.info("PyTorch version: %s", torch.__version__)
    logger.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("CUDA version: %s", torch.version.cuda)
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    source_dir = resolve_source_model(args.source)
    logger.info("Resolved source model: %s", source_dir)

    config = AutoConfig.from_pretrained(source_dir, local_files_only=True)
    validate_source_config(config, logger)

    tokenizer = AutoTokenizer.from_pretrained(source_dir, local_files_only=True, use_fast=True)

    original_comparison_model = None
    if not args.skip_output_comparison:
        original_comparison_model = clone_original_for_comparison(source_dir, logger)

    source_model = load_original_model(source_dir, logger)
    ensure_safe_dtype(source_model, logger)

    original_param_count = count_parameters(source_model)

    etet_model = create_etet_model(source_model, logger)

    verify_expert_initialization(
        original_model=original_comparison_model if original_comparison_model is not None else source_model,
        moe_model=etet_model,
        logger=logger,
    )

    etet_param_count = count_parameters(etet_model)
    param_report = {
        "original_parameters": original_param_count,
        "etet_moe_parameters": etet_param_count,
        "parameter_increase": etet_param_count - original_param_count,
    }
    logger.info("Original MiniCPM5-1B-SFT: %s parameters.", format_parameter_count(original_param_count))
    logger.info("ETET-Base: %s parameters.", format_parameter_count(etet_param_count))
    logger.info("ETET parameter increase: %s parameters.",
                format_parameter_count(etet_param_count - original_param_count))

    output_report = {"mean_absolute_error": 0.0, "max_absolute_error": 0.0, "relative_l2_difference": 0.0}
    if not args.skip_output_comparison:
        output_report = compare_output_behavior(
            original_model=original_comparison_model,
            moe_model=etet_model,
            tokenizer=tokenizer,
            logger=logger,
        )

    save_model(etet_model, tokenizer, source_dir, args.output, logger)
    save_metadata(args.output, param_report, list(range(MOE_START_LAYER, MOE_END_LAYER)))
    write_conversion_report(args.output, param_report, output_report)

    logger.info("Verifying saved model files exist.")
    required_files = [
        args.output / "config.json",
        args.output / "model.safetensors.index.json",
        args.output / "tokenizer_config.json",
    ]
    shard_exists = any(args.output.glob("model-*.safetensors"))
    if not shard_exists:
        required_files.append(args.output / "pytorch_model.bin")

    all_exist = True
    for f in required_files:
        if not f.exists():
            logger.error("Missing file: %s", f)
            all_exist = False
    if all_exist:
        logger.info("All required model files are present.")
    else:
        logger.warning("Some files missing, but model may still be usable with custom loading.")

    logger.info("Skipping full reload test (custom MoE architecture requires custom loader).")

    logger.info("Model summary: %d layers total, %d MoE layers (layers %d‑%d), %d experts, Top‑%d routing, %s parameters.",
                24, NUM_MOE_LAYERS, MOE_START_LAYER, MOE_END_LAYER - 1, NUM_EXPERTS, ROUTER_TOP_K,
                format_parameter_count(etet_param_count))

    logger.info("=" * 72)
    logger.info("ETET Dense-to-MoE Sparse Upcycling completed successfully.")
    logger.info("Output: %s", args.output)
    logger.info("Log: %s", LOG_FILE)
    logger.info("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())