# ETETMoE

[English](README.md) | [简体中文](README_zh-CN.md) | [繁體中文](README_zh-TW.md)

**ETETMoE** is the multimodal MoE architecture and pipeline built by RheoEcho based on the idea of **Dense-to-MoE Sparse Upcycling**. It is not only committed to running models on edge devices, but also to letting low-compute devices successfully run the entire training pipeline end to end. Its efficient MoE architecture brings high-performance multimodal large models to every device.

# Current Version
## Architecture Version
> 1.0

> ETETMoE_LLAMA
## Released Model Family
1. ETET-1.0-24E
- [ETET-1.0-24E-1.8B-A1B-Preview](https://huggingface.co/RheoEcho/ETET-1.0-24E-1.8B-A1B-Preview) 
## Current Version Notes
- It converts a small dense causal language model (MiniCPM5-1B-SFT) into a **dense + Mixture-of-Experts (MoE)** hybrid architecture. With only a little extra training, it attaches a SigLIP-HD vision encoder to support multimodal capabilities.
- Experiments show the whole process runs on a single NVIDIA Geforce RTX 4060 Laptop GPU.

---

## Repository Structure

| Path | Purpose |
| --- | --- |
| `env.py` | Shared environment, constants, `get_logger`, hardware/config helpers |
| `download.py` | Download the base model (MiniCPM5-1B-SFT) and SigLIP-HD checkpoints |
| `upcycle.py` | **Dense-to-MoE Sparse Upcycling**: build the ETET-MoE weight layout |
| `train.py` | Stage 2 — text fine-tuning to auto-differentiate expert roles |
| `visiontrain.py` | Stage 3 — multimodal alignment |
| `benchmark.py` | Evaluation framework (MMLU 5-shot, text path) |
| `etet_id_datasets.py` | Identity / instruction-tuning dataset assembly |
| `infer_tui.py` | Textual TUI chat: supports text and multimodal (VL) ETET as well as standard Llama models |
| `models/` | Local model checkpoints (one subdirectory per model) |
| `output/` | Training logs / outputs |
| `test/` | Test resources |

---

## Pipeline

```
MiniCPM5-1B-SFT ──upcycle.py──▶ ETETMoE_LLAMA (layers 0-15 dense + layers 16-23 MoE)
        │
        ├─ train.py ───────▶ ETET-Expert   (text SFT on the upcycled model to differentiate experts)
        │
        └─ visiontrain.py ─▶ ETETVL   (attach SigLIP-HD + Connector, do VL SFT)
```

---

## Quick Start

```bash
git clone https://github.com/Rheoecho-Studio/ETETMoE
cd ETETMoE
python env.py

# 1) Get the base checkpoint
python download.py

# 2) Build the ETET-MoE layout from the dense base model
python upcycle.py

# 3) Text fine-tuning (ETETLM)
python train.py

# 4) Multimodal alignment (ETETVL)
python visiontrain.py

# 5) Chat / evaluate
python infer_tui.py 
python benchmark.py
```

> For the full set of CLI options per script (batch size, learning rate, epochs, image size, number of experts, MoE start/end layers, etc.), see the detailed code inside each script.

---

## Environment Requirements

- Python ≥ 3.11
- PyTorch ≥ 2.x with CUDA (bfloat16)
- `transformers`, `safetensors`, `PIL`, `rich`, `textual`, fast tokenizer
- bfloat16 inference needs roughly 6–8 GB VRAM; training needs more (Preview scale runs on a single consumer GPU)

---

## Evaluation Results

Benchmark scores for the released model (MMLU, etc.) are maintained on the [Hugging Face model card](https://huggingface.co/RheoEcho/ETET-1.0-24E-1.8B-A1B-Preview). This repository does not duplicate them, in order to stay focused on the architecture and training code.

---

## License

- Code: AGPL v3.
- Model weights: Apache-2.0.
- The base model MiniCPM5-1B-SFT, SigLIP-HD, and all datasets used are governed by their respective licenses.

## Acknowledgements

- Based on [MiniCPM5-1B-SFT](https://huggingface.co/openbmb/MiniCPM5-1B) (OpenBMB)
- Vision encoder: [SigLIP-HD](https://huggingface.co/LiheYoung/SigLIP-HD) (LiheYoung).
- The upcycling concept is inspired by *Sparse Upcycling* (Komatsuzaki et al., 2023).

# Copyright Notice
- The "ETET" and "RheoEcho" marks are owned by RheoEcho. No third party may use them without RheoEcho's authorization.
- All files and code in this repository are licensed under AGPL v3. Even if a file does not explicitly state this, it remains legally binding; infringements will be pursued.
- Test images are provided by mirn; copyright belongs to him.
- RheoEcho 2026
