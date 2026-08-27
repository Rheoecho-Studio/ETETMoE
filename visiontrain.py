from __future__ import annotations
import argparse, gc, json, logging, math, os, random, shutil, sys, time, zipfile
from copy import deepcopy
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoConfig, AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file

torch.set_num_threads(16)
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
DATASETS_DIR = PROJECT_ROOT / "datasets"
OUTPUT_DIR = PROJECT_ROOT / "output"
TMP_DIR = MODELS_DIR / "tmp"

NUM_EXPERTS = 3
TOP_K = 1
MOE_START_LAYER = 16
MOE_END_LAYER = 23
NUM_TOTAL_LAYERS = 24
MOE_AUX_LOSS_COEF = 0.01
IMAGE_SIZE = 512
IMAGE_TOKEN = "<image>"
PREVIEW_SAMPLES = 8000      
FULL_SAMPLES = 100000       
MAX_TEXT_TOKENS = 512
TRAIN_EPOCHS = 1
TRAIN_LR = 1e-3
TRAIN_GRAD_ACCUM = 16
BATCH_SIZE = 1
SEED = 42
PROGRESS_FILE = TMP_DIR / "etet_mm_progress.json"
FEATURE_PROGRESS_EVERY = 500
STAGE_B_EPOCHS = 1
STAGE_B_LR = 2e-5
STAGE_B_GRAD_ACCUM = 16
STAGE_B_SAMPLES_PREVIEW = 5000 
STAGE_B_SAMPLES_FULL = 10000 
UNFREEZE_MOE_LAYERS = [22, 23]

def setup_logging():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("etet_visiontrain")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    fh = logging.FileHandler(OUTPUT_DIR / "visiontrain_output.log", mode="a", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger
LOGGER = setup_logging()

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def get_gpu_memory_gb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1024 ** 3

def format_duration(seconds):
    if not math.isfinite(seconds):
        return "--:--:--"
    return str(timedelta(seconds=max(0, int(seconds))))

def log_environment():
    LOGGER.info("=" * 72)
    LOGGER.info("ETET Vision Connector Training (LLM & Vision Frozen)")
    LOGGER.info("Python: %s", sys.version.replace("\n", " "))
    LOGGER.info("PyTorch: %s", torch.__version__)
    LOGGER.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        LOGGER.info("GPU: %s", torch.cuda.get_device_name(0))
        LOGGER.info("GPU memory: %.2f GB", get_gpu_memory_gb())
    LOGGER.info("=" * 72)

def safe_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return ""

def release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

def check_disk_space(needed_bytes, label):
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(str(TMP_DIR))
    free_gb = usage.free / 1024 ** 3
    need_gb = needed_bytes / 1024 ** 3
    if usage.free < needed_bytes:
        raise RuntimeError(f"Insufficient disk space for {label}: need {need_gb:.1f} GB, only {free_gb:.1f} GB free.")
    LOGGER.info("Disk space check for %s: need %.1f GB, free %.1f GB.", label, need_gb, free_gb)

def choose_mode_and_samples():
    print("\n" + "=" * 72)
    print("Select ETET training configuration")
    print("=" * 72)
    print("1. Preview text model + 8K samples (Stage A) / 5K (Stage B)")
    print("2. Full text model + 100K samples (Stage A) / 10K (Stage B)")
    print("3. Full text model + 8K samples (Stage A) / 8K (Stage B)")
    print("=" * 72)
    while True:
        v = input("Enter 1, 2, or 3: ").strip()
        if v == "1":
            mode, samples = "Preview", PREVIEW_SAMPLES
            break
        elif v == "2":
            mode, samples = "Full", FULL_SAMPLES
            break
        elif v == "3":
            mode, samples = "Full", PREVIEW_SAMPLES
            break
        print("Invalid selection. Enter 1, 2, or 3.")
    LOGGER.info("Selected text model: %s", mode)
    LOGGER.info("Selected LLaVA-Pretrain sample count: %d", samples)
    return mode, samples

def ask_stage_b():
    print("\n" + "=" * 72)
    print("Stage B: Joint Multimodal Fine-tuning (unfreezes last 2 MoE layers)")
    print("This can improve image understanding but may slightly affect text abilities.")
    print("=" * 72)
    while True:
        ans = input("Run Stage B? (y/N): ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no", ""):
            return False
        print("Please enter y or n.")

def get_text_model_dir(mode):
    return MODELS_DIR / ("ETET_Preview_text" if mode == "Preview" else "ETET_Full_text")

def get_final_output_dir(mode, samples):
    if mode == "Preview" and samples == PREVIEW_SAMPLES:
        return MODELS_DIR / "ETET_Preview_VL_Preview"
    if mode == "Full" and samples == FULL_SAMPLES:
        return MODELS_DIR / "ETET_VL"
    if mode == "Full" and samples == PREVIEW_SAMPLES:
        return MODELS_DIR / "ETET_VL_Preview"
    return MODELS_DIR / f"ETET_{mode}_VL_{samples}"

def find_siglip_dir():
    candidates = [MODELS_DIR / "SigLIP-HD", MODELS_DIR / "LiheYoung" / "SigLIP-HD", MODELS_DIR / "siglip-so400m-patch14-384"]
    for c in candidates:
        if c.exists():
            return c
    if MODELS_DIR.exists():
        for p in sorted(MODELS_DIR.iterdir()):
            if p.is_dir() and "siglip" in p.name.lower():
                return p
            nested = p / "SigLIP-HD"
            if p.is_dir() and nested.exists():
                return nested
    raise FileNotFoundError("SigLIP-HD vision encoder not found.")

# MoE MLP & helpers
class ETETMoEMLP(nn.Module):
    def __init__(self, dense_mlp, hidden_size, num_experts=NUM_EXPERTS, top_k=TOP_K):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([deepcopy(dense_mlp) for _ in range(num_experts)])
        self.router = nn.ModuleDict({"linear": nn.Linear(hidden_size, num_experts, bias=False)})
        self.last_aux_loss = torch.tensor(0.0)

    def forward(self, hidden_states):
        orig_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, orig_shape[-1])
        router_logits = self.router["linear"](flat)
        router_probs = torch.softmax(router_logits.float(), dim=-1)
        top_values, top_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        selected = top_indices[:, 0]
        output = torch.zeros_like(flat)
        dispatch, prob = [], []
        for expert_idx, expert in enumerate(self.experts):
            mask = selected == expert_idx
            count = mask.sum()
            dispatch.append(count.float() / max(1, flat.shape[0]))
            prob.append(router_probs[:, expert_idx].mean())
            if count.item() == 0:
                continue
            expert_out = expert(flat[mask])
            weight = top_values[mask, 0].to(expert_out.dtype)
            output[mask] += expert_out * weight.unsqueeze(-1)
        self.last_aux_loss = self.num_experts * torch.sum(torch.stack(dispatch) * torch.stack(prob))
        return output.reshape(orig_shape)

def find_model_weight_files(model_dir):
    files = []
    for p in model_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix == ".safetensors" or p.name in ("pytorch_model.bin",):
            files.append(p)
    return sorted(files)

def load_model_state_dict(model_dir):
    weight_files = find_model_weight_files(model_dir)
    if not weight_files:
        raise FileNotFoundError(f"No weight files under {model_dir}")
    state_dict = {}
    safetensor_files = [p for p in weight_files if p.suffix == ".safetensors"]
    if safetensor_files:
        for p in safetensor_files:
            LOGGER.info("Loading weights: %s", p.name)
            state_dict.update(safe_load_file(str(p), device="cpu"))
        return state_dict
    for p in weight_files:
        if p.name.endswith(".bin"):
            shard = torch.load(p, map_location="cpu", weights_only=True)
            if isinstance(shard, dict):
                state_dict.update(shard)
    if not state_dict:
        raise RuntimeError("Failed to load weights")
    return state_dict

def infer_moe_layers(state_dict):
    import re
    layers = set()
    pattern = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.")
    for key in state_dict:
        m = pattern.search(key)
        if m:
            layers.add(int(m.group(1)))
    return sorted(layers)

def create_etet_text_model(model_dir, dtype):
    import types
    from transformers import AutoModelForCausalLM
    LOGGER.info("Loading ETET text model from: %s", model_dir)
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    if getattr(config, "num_hidden_layers", None) != NUM_TOTAL_LAYERS:
        raise RuntimeError(f"Expected {NUM_TOTAL_LAYERS} layers, got {config.num_hidden_layers}")
    state_dict = load_model_state_dict(model_dir)
    moe_layers = infer_moe_layers(state_dict)
    expected = list(range(MOE_START_LAYER, MOE_END_LAYER + 1))
    if moe_layers != expected:
        raise RuntimeError(f"MoE layers {moe_layers} != expected {expected}")
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    hidden_size = config.hidden_size
    for layer_idx in expected:
        layer = model.model.layers[layer_idx]
        layer.mlp = ETETMoEMLP(layer.mlp, hidden_size, NUM_EXPERTS, TOP_K)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    relevant_missing = [k for k in missing if any(k.startswith(f"model.layers.{l}.mlp") for l in expected)]
    relevant_unexpected = [k for k in unexpected if any(k.startswith(f"model.layers.{l}.mlp") for l in expected)]
    if relevant_missing:
        raise RuntimeError(f"Missing ETET MoE parameters: {relevant_missing[:10]}")
    if relevant_unexpected:
        raise RuntimeError(f"Unexpected ETET MoE parameters: {relevant_unexpected[:10]}")
    model = model.to(dtype)
    model.config.use_cache = False
    original_forward = model.forward
    def etet_forward(self, *args, **kwargs):
        output = original_forward(*args, **kwargs)
        aux_losses = []
        for layer_idx in expected:
            mlp = self.model.layers[layer_idx].mlp
            if isinstance(mlp, ETETMoEMLP):
                aux_losses.append(mlp.last_aux_loss)
        if aux_losses:
            aux_loss = torch.stack([loss.to(output.logits.dtype) for loss in aux_losses]).mean()
            if getattr(output, "loss", None) is not None:
                output.loss = output.loss + MOE_AUX_LOSS_COEF * aux_loss
        return output
    model.forward = types.MethodType(etet_forward, model)
    LOGGER.info("ETET text model loaded with %d MoE layers.", len(expected))
    return model

def _extract_hidden(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor) and item.dim() == 3:
                return item
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if hasattr(output, "hidden_states") and output.hidden_states is not None:
        return output.hidden_states[-1]
    raise RuntimeError("Cannot extract hidden states")

class ETETVisionTower(nn.Module):
    def __init__(self, model, patch_size=14):
        super().__init__()
        self.model = model
        self.patch_size = patch_size
        self.expected_patches = ((IMAGE_SIZE - patch_size) // patch_size + 1) ** 2
    def forward(self, pixel_values):
        try:
            output = self.model(pixel_values=pixel_values, interpolate_pos_encoding=True)
        except TypeError:
            output = self.model(pixel_values=pixel_values)
        except Exception:
            output = self.model(pixel_values=pixel_values)
        hidden = _extract_hidden(output)
        if hidden.dim() == 3 and hidden.shape[1] == self.expected_patches + 1:
            hidden = hidden[:, 1:]
        return hidden
    @property
    def hidden_size(self):
        cfg = getattr(self.model, "config", None)
        if cfg is not None and hasattr(cfg, "hidden_size"):
            return cfg.hidden_size
        return 1152

def load_vision_tower(vision_dir):
    LOGGER.info("Loading SigLIP-HD vision encoder from: %s", vision_dir)
    config = AutoConfig.from_pretrained(vision_dir, local_files_only=True)
    patch_size = getattr(config, "patch_size", 14)
    model_type = str(getattr(config, "model_type", "")).lower()
    if "siglip" in model_type:
        from transformers import SiglipVisionModel
        tower_model = SiglipVisionModel.from_pretrained(vision_dir, local_files_only=True)
    else:
        full_model = AutoModel.from_pretrained(vision_dir, local_files_only=True)
        tower_model = full_model
        for attr in ("vision_model", "model", "vision_tower"):
            if hasattr(full_model, attr):
                tower_model = getattr(full_model, attr)
                break
    LOGGER.info("SigLIP-HD loaded via transformers.")
    tower = ETETVisionTower(tower_model, patch_size=patch_size)
    return tower

def build_image_processor(vision_dir):
    try:
        from transformers import AutoImageProcessor
        proc = AutoImageProcessor.from_pretrained(vision_dir, local_files_only=True)
        proc.size = {"height": IMAGE_SIZE, "width": IMAGE_SIZE}
        if hasattr(proc, "crop_size"):
            proc.crop_size = {"height": IMAGE_SIZE, "width": IMAGE_SIZE}
        LOGGER.info("Image processor loaded (resized to %dx%d).", IMAGE_SIZE, IMAGE_SIZE)
        return proc
    except Exception as exc:
        LOGGER.warning("AutoImageProcessor unavailable (%s); using manual normalization.", exc)
        return None

class ETETVisionConnector(nn.Module):
    def __init__(self, vision_hidden_size, lm_hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vision_hidden_size, lm_hidden_size),
            nn.GELU(),
            nn.Linear(lm_hidden_size, lm_hidden_size),
            nn.LayerNorm(lm_hidden_size, eps=1e-6),
        )
        with torch.no_grad():
            self.net[-1].weight.fill_(1.0 / math.sqrt(lm_hidden_size))
            self.net[-1].bias.zero_()
    def forward(self, image_features):
        return self.net(image_features)

class ETETMultimodal(nn.Module):
    def __init__(self, language_model, vision_tower, connector, image_token_id):
        super().__init__()
        self.language_model = language_model
        self.vision_tower = vision_tower
        self.connector = connector
        self.image_token_id = image_token_id
    def forward(self, input_ids, attention_mask, labels, pixel_values=None, image_features=None):
        if image_features is not None:
            connector_device = next(self.connector.parameters()).device
            image_features = image_features.to(connector_device)
            projected = self.connector(image_features)
        else:
            tower_device = next(self.vision_tower.parameters()).device
            pixel_values = pixel_values.to(tower_device)
            with torch.no_grad():
                image_features = self.vision_tower(pixel_values)
            connector_device = next(self.connector.parameters()).device
            image_features = image_features.to(connector_device)
            projected = self.connector(image_features)
        lm_device = next(self.language_model.parameters()).device
        embed_layer = self.language_model.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids).to(lm_device).clone()
        image_mask = input_ids == self.image_token_id
        batch_size = input_ids.shape[0]
        for b in range(batch_size):
            positions = image_mask[b].nonzero(as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            n = min(positions.numel(), projected.shape[1])
            inputs_embeds[b, positions[:n]] = projected[b, :n].to(inputs_embeds.dtype)
        if self.training and not inputs_embeds.requires_grad:
            inputs_embeds.requires_grad_(True)
        attention_mask = attention_mask.to(lm_device)
        labels = labels.to(lm_device)
        outputs = self.language_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        return outputs

class DiskFeatureCache:
    def __init__(self, path, shape, valid):
        self.path = path; self.shape = shape; self.valid = valid; self._mmap = None
    def _ensure_loaded(self):
        if self._mmap is None:
            self._mmap = np.memmap(str(self.path), dtype=np.float16, mode="r", shape=self.shape)
    def get(self, idx):
        if idx not in self.valid:
            return None
        self._ensure_loaded()
        arr = self._mmap[idx]
        return torch.from_numpy(arr.copy()).to(torch.bfloat16)
    def close(self):
        if self._mmap is not None:
            del self._mmap; self._mmap = None
        gc.collect()
        try:
            self.path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            self.path.with_suffix(".progress.json").unlink(missing_ok=True)
        except Exception:
            pass

def discover_data_files(root):
    return [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in (".json", ".jsonl")]

def load_llava_records(samples):
    root = DATASETS_DIR / "LLaVA-Pretrain"
    if not root.exists():
        raise FileNotFoundError(f"LLaVA-Pretrain not found: {root}")
    files = discover_data_files(root)
    LOGGER.info("Discovered %d candidate LLaVA-Pretrain data files.", len(files))
    records = []
    for path in files:
        if "_meta" in path.name.lower() or "metadata" in path.name.lower():
            continue
        rel = str(path.relative_to(root)).lower()
        if "test" in rel or "val" in rel:
            continue
        loaded = []
        try:
            if path.suffix.lower() == ".jsonl":
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            if isinstance(rec, dict):
                                loaded.append(rec)
                        except json.JSONDecodeError:
                            continue
            else:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        loaded = [r for r in data if isinstance(r, dict)]
        except Exception as exc:
            LOGGER.warning("Failed to parse %s: %s", path.name, exc)
        LOGGER.info("Loaded %d records from %s", len(loaded), path.name)
        records.extend(loaded)
    if not records:
        raise RuntimeError("No usable records found.")
    if len(records) > samples:
        rng = random.Random(SEED)
        indices = list(range(len(records)))
        rng.shuffle(indices)
        records = [records[i] for i in indices[:samples]]
        LOGGER.info("Sampled %d records (target %d).", len(records), samples)
    else:
        LOGGER.info("Only %d records available; using all.", len(records))
    return records

def ensure_images_extracted(records, image_root):
    images_dir = image_root / "images"
    if images_dir.exists() and any(images_dir.rglob("*.jpg")):
        LOGGER.info("Images already extracted.")
        return True
    zip_path = image_root / "images.zip"
    if not zip_path.exists():
        LOGGER.error("No images.zip found.")
        return False
    needed = set()
    for rec in records:
        img_name = safe_text(rec.get("image"))
        if img_name:
            needed.add(img_name.lower())
    LOGGER.info("Need to extract %d unique images from zip.", len(needed))
    images_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        all_members = zf.namelist()
        to_extract = []
        for member in all_members:
            if member.endswith("/"):
                continue
            m = member.lower()
            target_rel = None
            if m in needed:
                target_rel = member
            elif m.startswith("images/") and m[7:] in needed:
                target_rel = member[7:]
            if target_rel is not None:
                to_extract.append((member, target_rel))
        LOGGER.info("Matched %d images in zip.", len(to_extract))
        if not to_extract:
            LOGGER.error("No matching images found.")
            return False
        for i, (zip_name, target_rel) in enumerate(to_extract):
            target_path = images_dir / target_rel
            if target_path.exists():
                extracted += 1
                continue
            try:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                data = zf.read(zip_name)
                target_path.write_bytes(data)
                extracted += 1
            except Exception:
                pass
            if (i + 1) % 5000 == 0:
                LOGGER.info("Extraction progress: %d/%d", i + 1, len(to_extract))
    LOGGER.info("Image extraction complete: %d extracted.", extracted)
    return extracted > 0

def normalize_turns(rec):
    convs = rec.get("conversations")
    turns = []
    if isinstance(convs, list):
        for t in convs:
            if not isinstance(t, dict):
                continue
            role = safe_text(t.get("from") or t.get("role")).lower()
            value = safe_text(t.get("value") or t.get("content"))
            if role in ("human", "user"):
                turns.append(("user", value))
            elif role in ("gpt", "assistant"):
                turns.append(("assistant", value))
    if not turns:
        caption = safe_text(rec.get("caption"))
        if caption:
            turns = [("user", "Describe this image."), ("assistant", caption)]
    return turns

class LLaVAPretrainDataset(Dataset):
    def __init__(self, records, tokenizer, image_token_id, num_image_tokens, image_root, processor, features_cache=None):
        self.records = records
        self.tokenizer = tokenizer
        self.image_token_id = image_token_id
        self.num_image_tokens = num_image_tokens
        self.image_root = Path(image_root)
        self.processor = processor
        self.features_cache = features_cache
        self.skipped_count = 0
        self._skip_log_count = 0
        self._template_exc_logged = 0
        self._template_type_logged = False
        self._template_preview_logged = False
        self._prefix_check_logged = False
        self._image_cache = {}
        if features_cache is None:
            self._build_image_cache()
            self._diagnostic_check()

    def __len__(self):
        return len(self.records)

    def _build_image_cache(self):
        search_dirs = [self.image_root / "images", self.image_root]
        for search_dir in search_dirs:
            if not search_dir.exists() or not search_dir.is_dir():
                continue
            for p in search_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                    if p.name not in self._image_cache:
                        self._image_cache[p.name] = p
                    rel = str(p.relative_to(search_dir)).replace("\\", "/")
                    if rel not in self._image_cache:
                        self._image_cache[rel] = p
        if len(self._image_cache) == 0:
            LOGGER.error("No image files found under %s.", self.image_root)

    def _resolve_image(self, image_name):
        if not image_name:
            return None
        norm = image_name.replace("\\", "/").lower()
        if norm in self._image_cache:
            return self._image_cache[norm]
        basename = Path(image_name).name
        if basename in self._image_cache:
            return self._image_cache[basename]
        stem = Path(basename).stem
        for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            key = stem + ext
            if key in self._image_cache:
                return self._image_cache[key]
        return None

    def _diagnostic_check(self):
        found = 0
        total = min(100, len(self.records))
        for i in range(total):
            rec = self.records[i]
            image_name = safe_text(rec.get("image"))
            if self._resolve_image(image_name) is not None:
                found += 1
        if total > 0:
            LOGGER.info("Image diagnostic: %d/%d images found (%.1f%%)", found, total, found/total*100)

    def _preprocess_image(self, img):
        if self.processor is not None:
            try:
                out = self.processor(images=img.convert("RGB"), return_tensors="pt")
                return out["pixel_values"].squeeze(0)
            except Exception:
                pass
        img = img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        tensor = (tensor - 0.5) / 0.5
        return tensor

    def _normalize_template_ids(self, raw):
        ids = raw
        for _ in range(4):
            if ids is None:
                return None
            if isinstance(ids, torch.Tensor):
                ids = ids.reshape(-1).tolist()
            elif isinstance(ids, np.ndarray):
                ids = ids.reshape(-1).tolist()
            elif isinstance(ids, dict):
                ids = ids.get("input_ids")
            elif not isinstance(ids, (list, tuple)) and hasattr(ids, "input_ids"):
                ids = getattr(ids, "input_ids")
            else:
                break
        if ids is None:
            return None
        if isinstance(ids, torch.Tensor):
            ids = ids.reshape(-1).tolist()
        if isinstance(ids, np.ndarray):
            ids = ids.reshape(-1).tolist()
        if not isinstance(ids, (list, tuple)) or len(ids) == 0:
            return None
        if isinstance(ids[0], torch.Tensor):
            ids = ids[0].reshape(-1).tolist()
        elif isinstance(ids[0], np.ndarray):
            ids = ids[0].reshape(-1).tolist()
        elif isinstance(ids[0], (list, tuple)):
            ids = list(ids[0])
        try:
            return [int(x) for x in ids]
        except (TypeError, ValueError):
            return None

    def _apply_template(self, messages, add_generation_prompt):
        raw = None
        try:
            raw = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_generation_prompt, enable_thinking=False)
        except Exception:
            try:
                raw = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_generation_prompt)
            except Exception:
                return None
        if not self._template_type_logged:
            LOGGER.info("apply_chat_template raw return type: %s", type(raw).__name__)
            self._template_type_logged = True
        return self._normalize_template_ids(raw)

    def _encode_text(self, rec):
        turns = normalize_turns(rec)
        if not turns:
            return None
        user_parts = [c for r, c in turns if r == "user"]
        asst_parts = [c for r, c in turns if r == "assistant"]
        if not asst_parts:
            return None
        user_text = "\n".join(user_parts)
        assistant_text = "\n".join(asst_parts)
        user_msg = {"role": "user", "content": IMAGE_TOKEN + "\n" + user_text}
        asst_msg = {"role": "assistant", "content": assistant_text}
        prompt_ids = self._apply_template([user_msg], add_generation_prompt=True)
        if prompt_ids is None:
            return None
        full_ids = self._apply_template([user_msg, asst_msg], add_generation_prompt=False)
        if full_ids is None:
            return None
        if self.image_token_id not in full_ids:
            return None
        if not self._prefix_check_logged:
            n = min(len(prompt_ids), len(full_ids))
            LOGGER.info("Template prefix check: %s", prompt_ids[:n] == full_ids[:n])
            self._prefix_check_logged = True
        if not self._template_preview_logged:
            try:
                LOGGER.info("Template sample decoded preview: %s", self.tokenizer.decode(full_ids[:60]))
                self._template_preview_logged = True
            except Exception:
                pass
        img_pos = full_ids.index(self.image_token_id)
        ids = full_ids[:img_pos] + [self.image_token_id] * self.num_image_tokens + full_ids[img_pos + 1:]
        prompt_expanded_len = len(prompt_ids) - 1 + self.num_image_tokens
        if prompt_expanded_len > len(ids):
            prompt_expanded_len = len(ids)
        labels = [-100] * prompt_expanded_len + ids[prompt_expanded_len:]
        max_len = MAX_TEXT_TOKENS + self.num_image_tokens + 64
        if len(ids) > max_len:
            ids = ids[:max_len]
            labels = labels[:max_len]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, index):
        rec = self.records[index]
        encoded = self._encode_text(rec)
        if encoded is None:
            self.skipped_count += 1
            return None
        input_ids, labels = encoded
        attention_mask = torch.ones_like(input_ids)
        if self.features_cache is not None:
            feats = self.features_cache.get(index)
            if feats is None:
                self.skipped_count += 1
                return None
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "image_features": feats}
        image_name = safe_text(rec.get("image"))
        image_path = self._resolve_image(image_name)
        if image_path is None:
            self.skipped_count += 1
            if self._skip_log_count < 5:
                LOGGER.warning("Image not found: %s", image_name)
                self._skip_log_count += 1
            return None
        try:
            img = Image.open(image_path)
            pixel_values = self._preprocess_image(img)
            img.close()
        except Exception as exc:
            self.skipped_count += 1
            if self._skip_log_count < 5:
                LOGGER.warning("Failed to load image %s: %s", image_path, exc)
                self._skip_log_count += 1
            return None
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "pixel_values": pixel_values}

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    maxlen = max(len(b["input_ids"]) for b in batch)
    pad_id = 0
    input_ids, attention_masks, labels = [], [], []
    for b in batch:
        n = len(b["input_ids"])
        pad = maxlen - n
        input_ids.append(torch.cat([b["input_ids"], torch.full((pad,), pad_id, dtype=torch.long)]))
        attention_masks.append(torch.cat([b["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
        labels.append(torch.cat([b["labels"], torch.full((pad,), -100, dtype=torch.long)]))
    result = {"input_ids": torch.stack(input_ids), "attention_mask": torch.stack(attention_masks), "labels": torch.stack(labels)}
    if "image_features" in batch[0]:
        result["image_features"] = torch.stack([b["image_features"] for b in batch])
    else:
        result["pixel_values"] = torch.stack([b["pixel_values"] for b in batch])
    return result

def load_tokenizer_with_image_token(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else "<|pad|>"
    tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
    image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    if not isinstance(image_token_id, int) or image_token_id < 0 or image_token_id >= len(tokenizer):
        raise RuntimeError(f"Failed to register image token {IMAGE_TOKEN}.")
    LOGGER.info("Image token %s registered with id %d", IMAGE_TOKEN, image_token_id)
    return tokenizer, image_token_id

def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False

def configure_trainable_params(mm):
    freeze_all(mm)
    for p in mm.connector.parameters():
        p.requires_grad = True
    trainable = sum(p.numel() for p in mm.parameters() if p.requires_grad)
    LOGGER.info("Trainable parameters (connector only): %d", trainable)

def configure_stage_b_trainable(mm, unfreeze_layers):
    freeze_all(mm)
    for p in mm.connector.parameters():
        p.requires_grad = True
    for layer_idx in unfreeze_layers:
        mlp = mm.language_model.model.layers[layer_idx].mlp
        if not isinstance(mlp, ETETMoEMLP):
            raise RuntimeError(f"Layer {layer_idx} is not ETETMoEMLP.")
        for expert in mlp.experts:
            for p in expert.parameters():
                p.requires_grad = True
        for p in mlp.router.parameters():
            p.requires_grad = True
    LOGGER.info("Stage B: unfroze MoE layers %s.", unfreeze_layers)
    trainable = sum(p.numel() for p in mm.parameters() if p.requires_grad)
    LOGGER.info("Stage B trainable parameters: %d", trainable)

def configure_gradient_checkpointing(mm):
    lm = mm.language_model
    if hasattr(lm, "gradient_checkpointing_enable"):
        try:
            lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            lm.gradient_checkpointing_enable()
        lm.config.use_cache = False
        LOGGER.info("Gradient checkpointing enabled.")

def _feature_cache_paths(cache_name, n):
    base = TMP_DIR / f"features_{cache_name}_{n}"
    return base.with_suffix(".dat"), base.with_suffix(".progress.json")

def _compress_ranges(indices):
    if not indices:
        return []
    ordered = sorted(indices)
    ranges = []
    start = prev = ordered[0]
    for x in ordered[1:]:
        if x == prev + 1:
            prev = x
            continue
        ranges.append([start, prev])
        start = prev = x
    ranges.append([start, prev])
    return ranges

def _decompress_ranges(ranges):
    out = set()
    for pair in ranges or []:
        try:
            a, b = int(pair[0]), int(pair[1])
        except Exception:
            continue
        out.update(range(a, b + 1))
    return out

def _load_feature_progress(progress_path, shape):
    if not progress_path.exists():
        return {"shape": list(shape), "valid": [], "count": 0}
    try:
        with progress_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if list(data.get("shape", [])) != list(shape):
            LOGGER.warning("Feature progress shape mismatch; starting fresh.")
            return {"shape": list(shape), "valid": [], "count": 0}
        return data
    except Exception:
        return {"shape": list(shape), "valid": [], "count": 0}

def _save_feature_progress(progress_path, shape, valid):
    try:
        tmp = progress_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump({"shape": list(shape), "valid": _compress_ranges(valid), "count": len(valid)}, f)
        tmp.replace(progress_path)
    except Exception:
        pass

def _scan_valid_prefix(cache_path, shape):
    try:
        mmap = np.memmap(str(cache_path), dtype=np.float16, mode="r", shape=shape)
        probe = np.asarray(mmap[:, 0, :8])
        nonzero = (probe != 0).any(axis=1)
        del mmap
        zero_idx = np.flatnonzero(~nonzero)
        if zero_idx.size == 0:
            return int(shape[0])
        return int(zero_idx[0])
    except Exception:
        return 0

def precompute_vision_features(mm, records, tokenizer, image_token_id, num_image_tokens, image_root, processor, device, dtype, cache_name):
    n = len(records)
    LOGGER.info("Pre-computing vision features for %d records (cache '%s')...", n, cache_name)
    mm.vision_tower.to(device)
    mm.vision_tower.eval()
    temp_ds = LLaVAPretrainDataset(records, tokenizer, image_token_id, num_image_tokens, image_root, processor)
    vision_hidden = mm.vision_tower.hidden_size
    needed_bytes = n * num_image_tokens * vision_hidden * 2
    check_disk_space(needed_bytes, f"feature cache '{cache_name}'")
    cache_path, progress_path = _feature_cache_paths(cache_name, n)
    shape = (n, num_image_tokens, vision_hidden)
    progress_data = _load_feature_progress(progress_path, shape)
    valid = _decompress_ranges(progress_data.get("valid") or [])
    file_ready = cache_path.exists() and cache_path.stat().st_size == needed_bytes
    if not file_ready:
        valid = set()
    if not valid and file_ready:
        adopted = _scan_valid_prefix(cache_path, shape)
        if adopted > 0:
            LOGGER.info("Adopted %d rows from existing file.", adopted)
            valid = set(range(adopted))
            _save_feature_progress(progress_path, shape, valid)
    if valid:
        LOGGER.info("Resuming: %d already done.", len(valid))
        mmap = np.memmap(str(cache_path), dtype=np.float16, mode="r+", shape=shape)
    else:
        progress_path.unlink(missing_ok=True)
        mmap = np.memmap(str(cache_path), dtype=np.float16, mode="w+", shape=shape)
    ok, skip = len(valid), 0
    since_checkpoint = 0
    try:
        with torch.no_grad():
            for idx in range(n):
                if idx in valid:
                    continue
                item = temp_ds[idx]
                if item is None:
                    skip += 1
                    continue
                pv = item["pixel_values"].unsqueeze(0).to(device, dtype=dtype)
                feats = mm.vision_tower(pv)
                mmap[idx] = feats.squeeze(0).cpu().float().numpy()
                valid.add(idx)
                ok += 1
                since_checkpoint += 1
                del pv, feats
                if since_checkpoint >= FEATURE_PROGRESS_EVERY:
                    _save_feature_progress(progress_path, shape, valid)
                    LOGGER.info("Pre-computed %d/%d (ok=%d skip=%d)", idx+1, n, ok, skip)
                    since_checkpoint = 0
                    release_memory()
    except KeyboardInterrupt:
        _save_feature_progress(progress_path, shape, valid)
        raise
    _save_feature_progress(progress_path, shape, valid)
    del mmap, temp_ds
    gc.collect()
    mm.vision_tower.to("cpu")
    release_memory()
    LOGGER.info("Pre-compute done: %d features, %d skipped.", ok, skip)
    if skip > 0 and ok == 0:
        raise RuntimeError("All records failed encoding.")
    return DiskFeatureCache(cache_path, shape, valid)

def load_progress():
    if not PROGRESS_FILE.exists():
        return {}
    try:
        with PROGRESS_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_progress(progress):
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    tmp = PROGRESS_FILE.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)
    tmp.replace(PROGRESS_FILE)

def connector_checkpoint_path(output_name):
    return TMP_DIR / f"etet_connector_{output_name.lower()}.safetensors"

def save_connector(mm, output_name):
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    path = connector_checkpoint_path(output_name)
    state = {k: v.detach().to("cpu", torch.float32) for k, v in mm.connector.state_dict().items()}
    safe_save_file(state, str(path))
    LOGGER.info("Connector saved: %s", path)

def load_connector(mm, output_name):
    path = connector_checkpoint_path(output_name)
    if not path.exists():
        return False
    state = safe_load_file(str(path), device="cpu")
    target_dtype = next(mm.connector.parameters()).dtype
    state = {k: v.to(target_dtype) for k, v in state.items()}
    mm.connector.load_state_dict(state)
    LOGGER.info("Connector loaded from: %s", path)
    return True

def validate_and_reset_progress(mode_progress, output_name, final_output):
    reset_any = False
    if mode_progress.get("trained"):
        result = mode_progress.get("result", {})
        if result.get("steps", 0) == 0:
            LOGGER.warning("Training marked complete but 0 steps; resetting.")
            mode_progress["trained"] = False
            cp = connector_checkpoint_path(output_name)
            if cp.exists():
                cp.unlink()
            reset_any = True
        else:
            cp = connector_checkpoint_path(output_name)
            if not cp.exists():
                LOGGER.warning("Training marked complete but connector checkpoint missing; resetting.")
                mode_progress["trained"] = False
                reset_any = True
    if reset_any and final_output.exists() and (final_output / "etet_multimodal_metadata.json").exists():
        LOGGER.warning("Removing stale output directory.")
        shutil.rmtree(final_output)
    return mode_progress

@dataclass
class TrainingResult:
    steps: int = 0
    tokens: int = 0
    mean_loss: float = 0.0
    elapsed_seconds: float = 0.0

def autocast_context(device):
    if device.type != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)

def log_progress(cur, total, loss, avg_loss, lr, tokens, elapsed, start):
    eta = elapsed / max(1, cur) * max(0, total - cur)
    tps = tokens / max(1e-6, (time.time() - start))
    LOGGER.info("Training | progress=%6.2f%% | step=%d/%d | loss=%.5f | avg_loss=%.5f | lr=%.3e | elapsed=%s | ETA=%s | tok/s=%.1f",
                cur / max(1, total) * 100, cur, total, loss, avg_loss, lr,
                format_duration(elapsed), format_duration(eta), tps)
    if torch.cuda.is_available():
        LOGGER.info("CUDA mem | allocated=%.2f GB | reserved=%.2f GB",
                     torch.cuda.memory_allocated() / 1024 ** 3,
                     torch.cuda.memory_reserved() / 1024 ** 3)

def train_model(mm, dataset, device, epochs, lr, grad_accum, stage_name="Stage A"):
    LOGGER.info("%s | device=%s | epochs=%d | lr=%.2e | grad_accum=%d | samples=%d",
                stage_name, device, epochs, lr, grad_accum, len(dataset))
    use_pin = device.type == "cuda" and getattr(dataset, "features_cache", None) is None
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                        pin_memory=use_pin, collate_fn=collate_fn)
    trainable = [p for p in mm.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters.")
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(loader) / grad_accum)
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(total_steps * 0.03))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    LOGGER.info("Optimizer steps/epoch=%d total=%d warmup=%d", steps_per_epoch, total_steps, warmup_steps)
    mm.train()
    result = TrainingResult()
    loss_sum = loss_count = cur_updates = 0
    skipped = 0
    stage_start = time.time()
    run_start = time.time()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        LOGGER.info("Starting epoch %d/%d", epoch+1, epochs)
        for batch_idx, batch in enumerate(loader):
            if batch is None:
                skipped += 1
                continue
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            try:
                with autocast_context(device):
                    outputs = mm(**batch)
                    loss = outputs.loss
                if not torch.isfinite(loss):
                    LOGGER.error("Non-finite loss: %s", loss.detach().float().item())
                    del outputs, loss
                    optimizer.zero_grad(set_to_none=True)
                    continue
                cur_loss = loss.detach().float().item()
                loss_sum += cur_loss
                loss_count += 1
                (loss / grad_accum).backward()
                result.tokens += int(batch["attention_mask"].sum().item())
                del outputs, loss
            except torch.cuda.OutOfMemoryError:
                LOGGER.warning("CUDA OOM; emptying cache.")
                del batch
                optimizer.zero_grad(set_to_none=True)
                release_memory()
                skipped += 1
                continue
            if (batch_idx + 1) % grad_accum == 0 or batch_idx == len(loader) - 1:
                grads = [p for p in mm.parameters() if p.requires_grad and p.grad is not None]
                if grads:
                    torch.nn.utils.clip_grad_norm_(grads, max_norm=5.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                cur_updates += 1
                result.steps = cur_updates
                result.mean_loss = loss_sum / max(1, loss_count)
                log_progress(cur_updates, total_steps, cur_loss, result.mean_loss,
                             scheduler.get_last_lr()[0], result.tokens, time.time() - stage_start, run_start)
                release_memory()
    if hasattr(dataset, "skipped_count") and dataset.skipped_count > 0:
        LOGGER.warning("Training skipped %d samples.", dataset.skipped_count)
    result.elapsed_seconds = time.time() - stage_start
    LOGGER.info("%s completed | mean_loss=%.5f | steps=%d | skipped_batches=%d",
                stage_name, result.mean_loss, result.steps, skipped)
    del optimizer, scheduler
    release_memory()
    return result

def _save_language_model_manual(model, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_sd = model.state_dict()
    state_dict = {}
    seen_ptrs = set()
    for k, v in raw_sd.items():
        v_cpu = v.detach().cpu()
        ptr = v_cpu.data_ptr()
        if ptr in seen_ptrs:
            state_dict[k] = v_cpu.clone()
        else:
            seen_ptrs.add(ptr)
            state_dict[k] = v_cpu
    safe_save_file(state_dict, str(output_dir / "model.safetensors"))
    if hasattr(model, "config") and hasattr(model.config, "save_pretrained"):
        model.config.save_pretrained(str(output_dir))
    LOGGER.info("Language model saved manually.")

def export_final_model(mm, tokenizer, image_token_id, num_image_tokens, output_dir, mode, meta_extra,
                       processor=None, source_vision_dir=None):
    LOGGER.info("Exporting final model to: %s", output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    connector_dir = output_dir / "vision_connector"
    connector_dir.mkdir(parents=True, exist_ok=True)
    safe_save_file({k: v.detach().to("cpu", torch.float32) for k, v in mm.connector.state_dict().items()},
                   str(connector_dir / "connector.safetensors"))
    vision_dir = output_dir / "vision_tower"
    tower_model = mm.vision_tower.model
    if hasattr(tower_model, "save_pretrained"):
        tower_model.save_pretrained(str(vision_dir), safe_serialization=True)
    else:
        vision_dir.mkdir(parents=True, exist_ok=True)
        safe_save_file({k: v.detach().to("cpu") for k, v in tower_model.state_dict().items()},
                       str(vision_dir / "vision_tower.safetensors"))
    if processor is not None:
        try:
            proc_size = {"height": IMAGE_SIZE, "width": IMAGE_SIZE}
            processor.size = proc_size
            if hasattr(processor, "crop_size"):
                processor.crop_size = proc_size
            processor.save_pretrained(str(vision_dir))
        except Exception:
            if source_vision_dir is not None:
                src_cfg = source_vision_dir / "preprocessor_config.json"
                if src_cfg.exists():
                    shutil.copy2(str(src_cfg), str(vision_dir / "preprocessor_config.json"))
    elif source_vision_dir is not None:
        src_cfg = source_vision_dir / "preprocessor_config.json"
        if src_cfg.exists():
            shutil.copy2(str(src_cfg), str(vision_dir / "preprocessor_config.json"))
    preprocessor_check = vision_dir / "preprocessor_config.json"
    if preprocessor_check.exists():
        LOGGER.info("preprocessor_config.json present.")
    else:
        LOGGER.warning("preprocessor_config.json MISSING!")
    language_dir = output_dir / "language_model"
    lm = mm.language_model
    if hasattr(lm, "save_pretrained"):
        try:
            lm.save_pretrained(str(language_dir), safe_serialization=True, max_shard_size="2GB")
        except RuntimeError:
            _save_language_model_manual(lm, language_dir)
    else:
        _save_language_model_manual(lm, language_dir)
    tokenizer.save_pretrained(str(output_dir))
    metadata = {
        "model_type": "ETET-Multimodal",
        "mode": mode,
        "architecture": "SigLIP-HD + ETETVisionConnector (with LayerNorm) + ETET-MoE",
        "image_size": IMAGE_SIZE,
        "image_token": IMAGE_TOKEN,
        "image_token_id": image_token_id,
        "num_image_tokens": num_image_tokens,
        "vision_encoder": "LiheYoung/SigLIP-HD",
        "connector": {"type": "mlp_2layer_gelu_layernorm", "vision_hidden": mm.vision_tower.hidden_size, "lm_hidden": mm.language_model.config.hidden_size},
        "language_model": {"num_hidden_layers": NUM_TOTAL_LAYERS, "moe_layers": list(range(MOE_START_LAYER, MOE_END_LAYER + 1)), "num_experts_per_layer": NUM_EXPERTS, "top_k": TOP_K},
    }
    metadata.update(meta_extra)
    with (output_dir / "etet_multimodal_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    LOGGER.info("Final multimodal model exported.")
    # Verify
    required = [output_dir / "etet_multimodal_metadata.json", output_dir / "vision_connector" / "connector.safetensors"]
    for f in required:
        if not f.exists():
            raise RuntimeError(f"Missing exported file: {f}")
    if not (output_dir / "language_model").exists() or not (output_dir / "vision_tower").exists():
        raise RuntimeError("Missing language_model or vision_tower directory.")
    if not (output_dir / "vision_tower" / "preprocessor_config.json").exists():
        raise RuntimeError("Missing preprocessor_config.json.")
    LOGGER.info("Export verification passed.")

def main():
    set_seed()
    log_environment()
    mode, samples = choose_mode_and_samples()
    text_model_dir = get_text_model_dir(mode)
    final_output = get_final_output_dir(mode, samples)
    output_name = final_output.name
    LOGGER.info("Text model source: %s", text_model_dir)
    LOGGER.info("Final output: %s", final_output)
    if not text_model_dir.exists():
        raise FileNotFoundError(f"Text model not found: {text_model_dir}")
    vision_dir = find_siglip_dir()
    LOGGER.info("Vision encoder directory: %s", vision_dir)
    device = get_device()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    LOGGER.info("Device: %s | dtype: %s", device, dtype)
    progress_key = output_name
    progress = load_progress()
    mode_progress = progress.get(progress_key, {"trained": False, "stage_b_trained": False})
    mode_progress = validate_and_reset_progress(mode_progress, output_name, final_output)
    progress[progress_key] = mode_progress
    save_progress(progress)
    if final_output.exists() and (final_output / "etet_multimodal_metadata.json").exists():
        LOGGER.info("Final model already exists; exiting.")
        return 0
    tokenizer, image_token_id = load_tokenizer_with_image_token(text_model_dir)
    language_model = create_etet_text_model(text_model_dir, dtype)
    if len(tokenizer) != language_model.get_input_embeddings().weight.shape[0]:
        LOGGER.info("Resizing embeddings.")
        language_model.resize_token_embeddings(len(tokenizer))
    vision_tower = load_vision_tower(vision_dir)
    vision_tower = vision_tower.to(dtype)
    vision_hidden = vision_tower.hidden_size
    lm_hidden = language_model.config.hidden_size
    connector = ETETVisionConnector(vision_hidden, lm_hidden).to(dtype)
    LOGGER.info("Connector: %d -> %d", vision_hidden, lm_hidden)
    mm = ETETMultimodal(language_model, vision_tower, connector, image_token_id)
    vision_tower.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=dtype, device=next(vision_tower.parameters()).device)
        feats = vision_tower(dummy)
        num_image_tokens = int(feats.shape[1])
        LOGGER.info("Number of image tokens per image: %d", num_image_tokens)
        with autocast_context(device):
            projected = connector(feats)
        text_norm = language_model.get_input_embeddings().weight.float().norm(dim=-1).mean().item()
        visual_norm = projected.float().norm(dim=-1).mean().item()
        LOGGER.info("Initial connector scale: text_norm=%.3f visual_norm=%.3f ratio=%.3f",
                    text_norm, visual_norm, visual_norm / max(text_norm, 1e-6))
        del projected
    del dummy, feats
    release_memory()
    processor = build_image_processor(vision_dir)
    image_root = DATASETS_DIR / "LLaVA-Pretrain"
    records = load_llava_records(samples)
    LOGGER.info("Ensuring images are available...")
    if not ensure_images_extracted(records, image_root):
        raise RuntimeError("No images available.")
    # Stage A
    if not mode_progress.get("trained", False):
        LOGGER.info("Pre-computing vision features for Stage A...")
        features_cache = precompute_vision_features(mm, records, tokenizer, image_token_id,
                                                    num_image_tokens, image_root, processor,
                                                    device, dtype, "vision_features")
    else:
        features_cache = None
    mm.vision_tower.to("cpu")
    mm.connector.to(device)
    mm.language_model.to(device)
    configure_gradient_checkpointing(mm)
    if not mode_progress.get("trained", False):
        LOGGER.info("=" * 72)
        LOGGER.info("STAGE A: Connector Alignment")
        LOGGER.info("Frozen: ETET-MoE + SigLIP-HD. Trainable: Connector only.")
        LOGGER.info("=" * 72)
        configure_trainable_params(mm)
        dataset_a = LLaVAPretrainDataset(records, tokenizer, image_token_id, num_image_tokens,
                                         image_root, processor, features_cache=features_cache)
        result_a = train_model(mm, dataset_a, device, TRAIN_EPOCHS, TRAIN_LR, TRAIN_GRAD_ACCUM, "Stage A")
        save_connector(mm, output_name)
        mode_progress["trained"] = True
        mode_progress["result"] = {"steps": result_a.steps, "tokens": result_a.tokens, "mean_loss": result_a.mean_loss}
        mode_progress["samples"] = samples
        progress[progress_key] = mode_progress
        save_progress(progress)
        release_memory()
        if features_cache is not None:
            features_cache.close()
            del features_cache
            gc.collect()
            release_memory()
    else:
        LOGGER.info("Stage A already completed; loading connector.")
        if not load_connector(mm, output_name):
            raise RuntimeError("Connector checkpoint missing.")
    # Stage B
    run_stage_b = ask_stage_b()
    stage_b_trained = mode_progress.get("stage_b_trained", False)
    if run_stage_b and not stage_b_trained:
        stage_b_samples = STAGE_B_SAMPLES_PREVIEW if mode == "Preview" else STAGE_B_SAMPLES_FULL
        LOGGER.info("Stage B will use %d samples", stage_b_samples)
        rng = random.Random(SEED + 10)
        stage_b_records = rng.sample(records, min(stage_b_samples, len(records)))
        if len(stage_b_records) < stage_b_samples:
            LOGGER.warning("Only %d records available for Stage B.", len(stage_b_records))
        LOGGER.info("Pre-computing vision features for Stage B...")
        features_cache_b = precompute_vision_features(mm, stage_b_records, tokenizer, image_token_id,
                                                      num_image_tokens, image_root, processor,
                                                      device, dtype, "stageB_features")
        mm.vision_tower.to("cpu")
        mm.connector.to(device)
        mm.language_model.to(device)
        LOGGER.info("=" * 72)
        LOGGER.info("STAGE B: Joint Fine-tuning (unfreeze layers %s)", UNFREEZE_MOE_LAYERS)
        LOGGER.info("=" * 72)
        configure_stage_b_trainable(mm, UNFREEZE_MOE_LAYERS)
        dataset_b = LLaVAPretrainDataset(stage_b_records, tokenizer, image_token_id, num_image_tokens,
                                         image_root, processor, features_cache=features_cache_b)
        result_b = train_model(mm, dataset_b, device, STAGE_B_EPOCHS, STAGE_B_LR, STAGE_B_GRAD_ACCUM, "Stage B")
        save_connector(mm, output_name)
        mode_progress["stage_b_trained"] = True
        mode_progress["stage_b_result"] = {"steps": result_b.steps, "tokens": result_b.tokens, "mean_loss": result_b.mean_loss}
        mode_progress["stage_b_samples"] = len(stage_b_records)
        progress[progress_key] = mode_progress
        save_progress(progress)
        release_memory()
        if features_cache_b is not None:
            features_cache_b.close()
            del features_cache_b
            gc.collect()
            release_memory()
    elif run_stage_b and stage_b_trained:
        LOGGER.info("Stage B already completed; connector loaded from earlier.")
    else:
        LOGGER.info("Stage B skipped.")
    # Export
    meta_extra = {
        "stage_a": {"epochs": TRAIN_EPOCHS, "lr": TRAIN_LR, "samples": samples},
        "stage_b": {"executed": run_stage_b,
                    "epochs": STAGE_B_EPOCHS if run_stage_b else 0,
                    "lr": STAGE_B_LR if run_stage_b else 0,
                    "unfrozen_layers": list(UNFREEZE_MOE_LAYERS) if run_stage_b else [],
                    "samples": len(stage_b_records) if run_stage_b else 0}
    }
    export_final_model(mm, tokenizer, image_token_id, num_image_tokens, final_output, mode, meta_extra,
                       processor=processor, source_vision_dir=vision_dir)
    LOGGER.info("=" * 72)
    LOGGER.info("ETET Vision training completed successfully.")
    LOGGER.info("Output: %s", final_output)
    LOGGER.info("=" * 72)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted by user.")
        sys.exit(130)
    except Exception as e:
        LOGGER.exception("Training failed: %s", e)
        sys.exit(1)