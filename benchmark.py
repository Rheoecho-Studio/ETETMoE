#!/usr/bin/env python3
"""benchmark.py - ETET MMLU / MMLU Pro 5-shot benchmark with interactive model selection."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM, GenerationConfig
from safetensors.torch import load_file
import pyarrow.parquet as pq

# ============================================================
# MMLU 57 subjects (full list)
# ============================================================
MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology",
    "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies",
    "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology",
    "us_foreign_policy", "virology", "world_religions"
]

# ============================================================
# MMLU Pro 14 categories (full list)
# ============================================================
MMLU_PRO_CATEGORIES = [
    "biology", "business", "chemistry", "computer science", "economics",
    "engineering", "health", "history", "law", "math", "other",
    "philosophy", "physics", "psychology"
]

# ============================================================
# ETET configuration (must match upcycle.py and infer_tui.py)
# ============================================================
NUM_EXPERTS = 3
ROUTER_TOP_K = 1
MOE_START_LAYER = 16
MOE_END_LAYER = 24
NUM_TOTAL_LAYERS = 24

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
DATASETS_DIR = PROJECT_ROOT / "datasets"

# ============================================================
# ETET MoE modules (must match upcycle.py structure)
# ============================================================
class Top1Router(nn.Module):
    def __init__(self, hidden_size, num_experts, dtype=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = ROUTER_TOP_K
        self.linear = nn.Linear(hidden_size, num_experts, bias=False, dtype=dtype)

    def forward(self, hidden_states):
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
    def __init__(self, dense_ffn, hidden_size, num_experts=NUM_EXPERTS, top_k=ROUTER_TOP_K):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        router_dtype = next(dense_ffn.parameters()).dtype
        self.router = Top1Router(hidden_size, num_experts, dtype=router_dtype)
        self.experts = nn.ModuleList([copy.deepcopy(dense_ffn) for _ in range(num_experts)])

    def forward(self, hidden_states):
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
# Model discovery and interactive selection
# ============================================================
def is_etet_model(model_dir: Path) -> bool:
    return "etet" in model_dir.name.lower()

def is_multimodal_model(model_dir: Path) -> bool:
    name = model_dir.name.lower()
    return "etet" in name and "vl" in name

def list_models() -> List[Path]:
    if not MODELS_DIR.exists():
        return []
    return sorted(
        [p for p in MODELS_DIR.iterdir() if p.is_dir() and p.name.lower() not in ("tmp", "gguf")],
        key=lambda p: p.name.lower(),
    )

def select_model_interactive(log_fn: Optional[Callable[[str], None]] = None) -> Optional[Path]:
    def log(msg):
        if log_fn:
            log_fn(msg)
        else:
            print(msg)

    models = list_models()
    if not models:
        log(f"No model directories found in: {MODELS_DIR}")
        return None
    log("")
    log("Available models:")
    for i, m in enumerate(models):
        if is_multimodal_model(m):
            tag = "ETET-Multimodal"
        elif is_etet_model(m):
            tag = "ETET-MoE"
        else:
            tag = "Llama"
        log(f"  [{i}] {m.name} [{tag}]")
    while True:
        choice = input(f"Select model [0-{len(models)-1}]: ").strip()
        try:
            idx = int(choice)
            if 0 <= idx < len(models):
                return models[idx]
        except ValueError:
            pass
        log("Invalid choice, try again.")

def select_benchmark() -> str:
    print("\nBenchmark suite:")
    print("  [1] MMLU       (57 subjects, ~14K questions, A-D)")
    print("  [2] MMLU Pro   (14 categories, ~12K questions, A-J)")
    while True:
        choice = input("Select [1-2]: ").strip()
        if choice in ("1", "2"):
            return "mmlu" if choice == "1" else "mmlu_pro"
        print("Invalid choice, try again.")

# ============================================================
# Checkpoint loading helpers
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
            raise FileNotFoundError("Missing shards:\n" + "\n".join(str(x) for x in missing))
        return files
    single_file = model_dir / "model.safetensors"
    if single_file.exists():
        return [single_file]
    files = sorted(model_dir.glob("*.safetensors"))
    if files:
        return files
    raise FileNotFoundError(f"No safetensors in: {model_dir}")

def load_safetensors_into_model(model, model_dir, log_fn=None):
    def log(msg):
        if log_fn:
            log_fn(msg)
    files = find_checkpoint_files(model_dir)
    log(f"Found {len(files)} safetensors shard(s).")
    model_keys = set(model.state_dict().keys())
    loaded_keys = set()
    unexpected = set()
    for i, f in enumerate(files, 1):
        log(f"Loading shard {i}/{len(files)}: {f.name}")
        shard = load_file(str(f), device="cpu")
        skeys = set(shard.keys())
        unexpected.update(skeys - model_keys)
        loaded_keys.update(skeys & model_keys)
        model.load_state_dict(shard, strict=False)
        del shard
        gc.collect()
    missing = model_keys - loaded_keys
    if missing:
        raise RuntimeError(f"Missing {len(missing)} tensors:\n" + "\n".join(sorted(missing)[:20]))
    if unexpected:
        raise RuntimeError(f"{len(unexpected)} unexpected tensors:\n" + "\n".join(sorted(unexpected)[:20]))
    log("Weights loaded.")

def validate_config(config):
    if getattr(config, "model_type", None) != "llama":
        raise ValueError(f"Expected model_type='llama', got {config.model_type!r}")
    if getattr(config, "num_hidden_layers", None) != NUM_TOTAL_LAYERS:
        raise ValueError(f"Expected {NUM_TOTAL_LAYERS} layers, got {config.num_hidden_layers}")

def load_selected_model(model_dir: Path, device, dtype, log_fn: Optional[Callable[[str], None]] = None):
    def log(msg):
        if log_fn:
            log_fn(msg)
        else:
            print(msg)

    etet = is_etet_model(model_dir)
    is_vl = is_multimodal_model(model_dir)
    language_dir = model_dir / "language_model" if is_vl else model_dir
    log(f"Model dir: {model_dir}")
    log(f"ETET MoE: {etet} | Multimodal: {is_vl}")
    log(f"Language model dir: {language_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)

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
                model.generation_config = GenerationConfig.from_pretrained(language_dir, local_files_only=True)
            except Exception:
                pass
        model.to(device=device, dtype=dtype)
        model.eval()
    else:
        log("Loading as standard LlamaForCausalLM.")
        model = LlamaForCausalLM.from_pretrained(
            language_dir, torch_dtype=dtype, local_files_only=True, low_cpu_mem_usage=True
        )
        model.to(device)
        model.eval()

    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        log(f"Resizing embeddings: {model.get_input_embeddings().weight.shape[0]} -> {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer

# ============================================================
# Shared evaluation utilities
# ============================================================
def check_single_token_choices(tokenizer, letters):
    base = "Answer:"
    base_ids = tokenizer.encode(base, add_special_tokens=False)
    choice_ids = []
    all_single = True
    for letter in letters:
        full = tokenizer.encode(base + " " + letter, add_special_tokens=False)
        cont = full[len(base_ids):]
        choice_ids.append(cont)
        if len(cont) != 1:
            all_single = False
    return all_single, choice_ids

def eval_single_token(model, prompt_ids, choice_token_ids, device):
    input_ids = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    with torch.inference_mode():
        logits = model(input_ids).logits
    next_logits = logits[0, -1, :]
    log_probs = torch.log_softmax(next_logits, dim=-1)
    return [log_probs[cti[0]].item() for cti in choice_token_ids]

def eval_full_loglikelihood(model, tokenizer, prompt, choices, device):
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    results = []
    for choice in choices:
        full_text = prompt + choice
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        cont_ids = full_ids[len(prompt_ids):]
        if not cont_ids:
            results.append(float('-inf'))
            continue
        input_ids = torch.tensor([full_ids], device=device, dtype=torch.long)
        with torch.inference_mode():
            logits = model(input_ids).logits
        log_probs = torch.log_softmax(logits[0], dim=-1)
        total = 0.0
        for i, tok_id in enumerate(cont_ids):
            pos = len(prompt_ids) + i - 1
            total += log_probs[pos, tok_id].item()
        results.append(total)
    return results

def read_parquet_to_dicts(file_path):
    table = pq.read_table(file_path)
    columns = table.column_names
    rows = []
    for i in range(table.num_rows):
        row = {col: table.column(col)[i].as_py() for col in columns}
        rows.append(row)
    return rows

# ============================================================
# MMLU data loading (skip download if cache is valid)
# ============================================================
def is_mmlu_cache_valid() -> bool:
    local_cache = DATASETS_DIR / "mmlu"
    if not local_cache.exists():
        return False
    for subj in MMLU_SUBJECTS:
        subj_dir = local_cache / subj
        if subj_dir.exists() and (subj_dir / "test-00000-of-00001.parquet").exists():
            return True
    return False

def load_mmlu_data(log_fn: Optional[Callable[[str], None]] = None):
    def log(msg):
        if log_fn:
            log_fn(msg)
        else:
            print(msg)

    local_cache = DATASETS_DIR / "mmlu"
    if is_mmlu_cache_valid():
        log(f"Local MMLU cache detected, skipping download: {local_cache}")
    else:
        log(f"Local cache incomplete, downloading MMLU to {local_cache}...")
        cmd = ["modelscope", "download", "--dataset", "cais/mmlu", "--local_dir", str(local_cache)]
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Download failed with code {result.returncode}")
        log("Download completed.")

    all_dev = []
    all_test = []
    for subj in MMLU_SUBJECTS:
        subj_dir = local_cache / subj
        dev_file = subj_dir / "dev-00000-of-00001.parquet"
        test_file = subj_dir / "test-00000-of-00001.parquet"
        try:
            if dev_file.exists():
                dev_records = read_parquet_to_dicts(dev_file)
                for rec in dev_records:
                    rec['subject'] = subj
                all_dev.extend(dev_records)
            if test_file.exists():
                test_records = read_parquet_to_dicts(test_file)
                for rec in test_records:
                    rec['subject'] = subj
                all_test.extend(test_records)
        except Exception as e:
            log(f"Error loading {subj}: {e}")
    log(f"Total loaded: {len(all_dev)} dev, {len(all_test)} test.")
    return all_dev, all_test

# ============================================================
# MMLU evaluation (Full 57 subjects)
# ============================================================
LETTERS = ['A', 'B', 'C', 'D']

def format_question_mmlu(question, choices, answer_idx=None):
    text = f"Question: {question}\n"
    for i, c in enumerate(choices):
        text += f"{LETTERS[i]}. {c}\n"
    if answer_idx is not None:
        text += f"Answer: {LETTERS[answer_idx]}"
    else:
        text += "Answer:"
    return text

def build_prompt_mmlu(subject, dev_examples, question, choices):
    subj_name = subject.replace('_', ' ')
    prompt = f"The following are multiple choice questions (with answers) about {subj_name}.\n\n"
    for ex in dev_examples:
        prompt += format_question_mmlu(ex['question'], ex['choices'], ex['answer']) + "\n\n"
    prompt += format_question_mmlu(question, choices)
    return prompt

def evaluate_mmlu(model, tokenizer, device, limit=0, num_fewshot=5, log_fn=None):
    def log(msg):
        if log_fn:
            log_fn(msg)
        else:
            print(msg)

    log("Loading MMLU dataset (57 subjects)...")
    dev, test = load_mmlu_data(log)
    dev_by_subject = {}
    for ex in dev:
        s = ex['subject']
        dev_by_subject.setdefault(s, []).append(ex)

    test_list = list(test)
    if limit > 0 and limit < len(test_list):
        random.seed(42)
        test_list = random.sample(test_list, limit)
        log(f"Subsampled to {limit} questions.")

    all_single, choice_token_ids = check_single_token_choices(tokenizer, LETTERS)
    log(f"Single-token choices: {all_single}")
    log(f"Total questions: {len(test_list)} | few-shot: {num_fewshot}")
    log("-" * 60)

    correct = 0
    total = 0
    subj_correct = {}
    subj_total = {}
    choices_str = [' A', ' B', ' C', ' D']
    t0 = time.time()
    last_log_time = t0

    for idx, ex in enumerate(test_list):
        subject = ex['subject']
        few = dev_by_subject.get(subject, [])[:num_fewshot]
        prompt = build_prompt_mmlu(subject, few, ex['question'], ex['choices'])
        if all_single:
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            lps = eval_single_token(model, prompt_ids, choice_token_ids, device)
        else:
            lps = eval_full_loglikelihood(model, tokenizer, prompt, choices_str, device)
        pred = lps.index(max(lps))
        ok = (pred == ex['answer'])
        if ok:
            correct += 1
        total += 1
        subj_correct[subject] = subj_correct.get(subject, 0) + (1 if ok else 0)
        subj_total[subject] = subj_total.get(subject, 0) + 1
        now = time.time()
        if (idx + 1) % 500 == 0 or now - last_log_time > 10:
            elapsed = now - t0
            acc = correct / total * 100
            eta = elapsed / (idx + 1) * (len(test_list) - idx - 1)
            log(f" [{idx+1:>6}/{len(test_list)}] acc={acc:.2f}% elapsed={elapsed:.0f}s eta={eta:.0f}s")
            last_log_time = now

    elapsed = time.time() - t0
    acc = correct / total * 100
    log("")
    log("=" * 60)
    log(f"MMLU {num_fewshot}-shot Results (Full 57 Subjects)")
    log("=" * 60)
    log(f"Overall: {correct}/{total} = {acc:.2f}%")
    log(f"Time: {elapsed:.1f}s ({total/elapsed:.1f} q/s)")
    log("-" * 60)
    log("Per-subject accuracy (sorted):")
    for subj in sorted(subj_correct.keys()):
        c = subj_correct[subj]
        t = subj_total[subj]
        log(f" {subj:45s} {c:4d}/{t:4d} = {c/t*100:6.2f}%")
    log("=" * 60)
    return acc

# ============================================================
# MMLU Pro data loading and evaluation
# ============================================================
LETTERS_PRO = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

def is_mmlu_pro_cache_valid() -> bool:
    local_cache = DATASETS_DIR / "mmlu_pro"
    if not local_cache.exists():
        return False
    test_files = list(local_cache.rglob("*test*.parquet"))
    return len(test_files) > 0

def load_mmlu_pro_data(log_fn: Optional[Callable[[str], None]] = None):
    def log(msg):
        if log_fn:
            log_fn(msg)
        else:
            print(msg)

    local_cache = DATASETS_DIR / "mmlu_pro"
    if is_mmlu_pro_cache_valid():
        log(f"Local MMLU Pro cache detected, skipping download: {local_cache}")
    else:
        log(f"Local cache incomplete, downloading MMLU Pro to {local_cache}...")
        cmd = ["modelscope", "download", "--dataset", "cais/mmlu_pro", "--local_dir", str(local_cache)]
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Download failed with code {result.returncode}")
        log("Download completed.")

    all_dev = []
    all_test = []
    for parquet_file in sorted(local_cache.rglob("*.parquet")):
        name = parquet_file.name.lower()
        try:
            records = read_parquet_to_dicts(parquet_file)
        except Exception as e:
            log(f"Error reading {parquet_file}: {e}")
            continue
        if "validation" in name or "dev" in name:
            all_dev.extend(records)
        elif "test" in name:
            all_test.extend(records)
    log(f"Total loaded: {len(all_dev)} dev, {len(all_test)} test.")
    return all_dev, all_test

def format_question_mmlu_pro(question, options, answer_letter=None):
    text = f"Question: {question}\n"
    for i, c in enumerate(options):
        text += f"{LETTERS_PRO[i]}. {c}\n"
    if answer_letter is not None:
        text += f"Answer: {answer_letter}"
    else:
        text += "Answer:"
    return text

def build_prompt_mmlu_pro(category, dev_examples, question, options):
    prompt = f"The following are multiple choice questions (with answers) about {category}.\n\n"
    for ex in dev_examples:
        prompt += format_question_mmlu_pro(ex['question'], ex['options'], ex.get('answer')) + "\n\n"
    prompt += format_question_mmlu_pro(question, options)
    return prompt

def evaluate_mmlu_pro(model, tokenizer, device, limit=0, num_fewshot=5, log_fn=None):
    def log(msg):
        if log_fn:
            log_fn(msg)
        else:
            print(msg)

    log("Loading MMLU Pro dataset...")
    dev, test = load_mmlu_pro_data(log)
    dev_by_category = {}
    for ex in dev:
        c = ex.get('category', 'other')
        dev_by_category.setdefault(c, []).append(ex)

    test_list = list(test)
    if limit > 0 and limit < len(test_list):
        random.seed(42)
        test_list = random.sample(test_list, limit)
        log(f"Subsampled to {limit} questions.")

    all_single, choice_token_ids = check_single_token_choices(tokenizer, LETTERS_PRO)
    log(f"Single-token choices (A-J): {all_single}")
    log(f"Total questions: {len(test_list)} | few-shot: {num_fewshot}")
    log("-" * 60)

    correct = 0
    total = 0
    cat_correct = {}
    cat_total = {}
    choices_str = [' ' + l for l in LETTERS_PRO]
    t0 = time.time()
    last_log_time = t0

    for idx, ex in enumerate(test_list):
        category = ex.get('category', 'other')
        few = dev_by_category.get(category, [])[:num_fewshot]
        question = ex['question']
        options = ex['options']
        answer_letter = ex.get('answer')
        answer_index = ex.get('answer_index')
        if answer_index is None:
            if answer_letter in LETTERS_PRO:
                answer_index = LETTERS_PRO.index(answer_letter)
            else:
                continue
        prompt = build_prompt_mmlu_pro(category, few, question, options)
        if all_single:
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            lps = eval_single_token(model, prompt_ids, choice_token_ids, device)
        else:
            lps = eval_full_loglikelihood(model, tokenizer, prompt, choices_str, device)
        pred = lps.index(max(lps))
        ok = (pred == answer_index)
        if ok:
            correct += 1
        total += 1
        cat_correct[category] = cat_correct.get(category, 0) + (1 if ok else 0)
        cat_total[category] = cat_total.get(category, 0) + 1
        now = time.time()
        if (idx + 1) % 500 == 0 or now - last_log_time > 10:
            elapsed = now - t0
            acc = correct / total * 100
            eta = elapsed / (idx + 1) * (len(test_list) - idx - 1)
            log(f" [{idx+1:>6}/{len(test_list)}] acc={acc:.2f}% elapsed={elapsed:.0f}s eta={eta:.0f}s")
            last_log_time = now

    elapsed = time.time() - t0
    acc = correct / total * 100
    log("")
    log("=" * 60)
    log(f"MMLU Pro {num_fewshot}-shot Results")
    log("=" * 60)
    log(f"Overall: {correct}/{total} = {acc:.2f}%")
    log(f"Time: {elapsed:.1f}s ({total/elapsed:.1f} q/s)")
    log("-" * 60)
    log("Per-category accuracy (sorted):")
    for cat in sorted(cat_correct.keys()):
        c = cat_correct[cat]
        t = cat_total[cat]
        log(f" {cat:30s} {c:4d}/{t:4d} = {c/t*100:6.2f}%")
    log("=" * 60)
    return acc

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="ETET MMLU / MMLU Pro 5-shot Benchmark")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="auto",
                        choices=["auto", "float32", "fp32", "float16", "fp16", "bfloat16", "bf16"])
    parser.add_argument("--limit", type=int, default=0, help="Subsample N questions (0 = all)")
    parser.add_argument("--fewshot", type=int, default=5)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device(args.device)

    if args.dtype == "auto":
        if device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
    else:
        m = {"float32": torch.float32, "fp32": torch.float32,
             "float16": torch.float16, "fp16": torch.float16,
             "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}
        dtype = m[args.dtype]

    logs: List[str] = []
    def my_log(msg):
        print(msg)
        logs.append(msg)

    my_log(f"Device: {device} | Dtype: {dtype}")
    my_log("")

    # Step 1: Interactive model selection
    model_dir = select_model_interactive(log_fn=my_log)
    if model_dir is None:
        return 1
    my_log(f"Selected model: {model_dir.name}")

    # Step 2: Interactive benchmark selection
    benchmark = select_benchmark()
    my_log(f"Selected benchmark: {benchmark}")

    # Step 3: Load model (ETET MoE if name contains "etet", else plain Llama)
    my_log("Loading model...")
    model, tokenizer = load_selected_model(model_dir, device, dtype, log_fn=my_log)
    params = sum(p.numel() for p in model.parameters())
    my_log(f"Parameters: {params/1e9:.2f}B")
    my_log("")

    # Step 4: Run the selected benchmark
    if benchmark == "mmlu":
        acc = evaluate_mmlu(model, tokenizer, device, limit=args.limit, num_fewshot=args.fewshot, log_fn=my_log)
        bench_name = "MMLU"
    else:
        acc = evaluate_mmlu_pro(model, tokenizer, device, limit=args.limit, num_fewshot=args.fewshot, log_fn=my_log)
        bench_name = "MMLU_Pro"

    my_log(f"\nFinal {bench_name} {args.fewshot}-shot accuracy: {acc:.2f}%")

    # Step 5: Save result to results/
    result_dir = Path("results")
    result_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_name = model_dir.name
    filename = result_dir / f"{bench_name.lower()}_{model_name}_{timestamp}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(logs))
    my_log(f"\nResult saved to: {filename}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
