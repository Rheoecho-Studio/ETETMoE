# ETETMoE

[English](README.md) | [简体中文](README_zh-CN.md) | [繁體中文](README_zh-TW.md)

**ETETMoE** 是RheoEcho参考**Dense-to-MoE Sparse Upcycling**思想打造的多模态MoE架构和流水线，不光致力于让边缘设备运行模型，还致力于让低算力设备成功跑通训练流水线全过程。利用高效的MoE架构让每一个设备用上高性能多模态大模型。

# 当前版本介绍
## 版本架构
> 1.0

> ETETMoE_LLAMA
## 产出模型系列
1. ETET-1.0-24E
- [ETET-1.0-24E-1.8B-A1B-Preview](https://huggingface.co/RheoEcho/ETET-1.0-24E-1.8B-A1B-Preview) 
## 当前版本介绍
- 它将一个小型稠密因果语言模型（MiniCPM5-1B-SFT）转换为 **稠密 + 混合专家（Mixture-of-Experts）** 混合架构模型，只需很少的额外训练，再挂接 SigLIP-HD 视觉编码器即可支持多模态。
- 经实验，本架构全部过程只需单卡 NVIDIA Geforce RTX 4060 Laptop 即可完成
---

## 仓库结构

| 路径 | 用途 |
| --- | --- |
| `env.py` | 共享环境、常量、`get_logger`、硬件/配置辅助 |
| `download.py` | 下载基础模型（MiniCPM5-1B-SFT）与 SigLIP-HD 检查点 |
| `upcycle.py` | **Dense-to-MoE Sparse Upcycling**：构建 ETET-MoE 权重布局 |
| `train.py` | 阶段 2——文本微调自动分化专家职能 |
| `visiontrain.py` | 阶段 3——多模态对齐 |
| `benchmark.py` | 评测框架（MMLU 5-shot，文本路径） |
| `etet_id_datasets.py` | 身份 / 指令微调数据集组装 |
| `infer_tui.py` | Textual TUI 聊天：支持文本与多模态（VL）ETET 和标准Llama模型 |
| `models/` | 本地模型检查点（每个模型一个子目录） |
| `output/` | 训练日志 / 输出 |
| `test/` | 测试资源 |


---

## 流水线

```
MiniCPM5-1B-SFT ──upcycle.py──▶ ETETMoE_LLAMA（第 0-15 层稠密 + 第 16-23 层 MoE）
        │
        ├─ train.py ───────▶ ETET-Expert   （在 upcycle 后的模型基础上做文本 SFT 进行专家分化）
        │
        └─ visiontrain.py ─▶ ETETVL   （挂接 SigLIP-HD + Connector，做 VL SFT）
```

---

## 快速开始

```bash
git clone https://github.com/Rheoecho-Studio/ETETMoE
cd ETETMoE
python env.py

# 1) 获取基础检查点
python download.py

# 2) 从稠密基础模型构建 ETET-MoE 布局
python upcycle.py

# 3) 文本微调（ETETLM）
python train.py

# 4) 多模态对齐（ETETVL）
python visiontrain.py

# 5) 聊天 / 评测
python infer_tui.py 
python benchmark.py
```

> 每个脚本的完整 CLI 选项（batch size、学习率、epoch、图像尺寸、专家数量、MoE 起止层等）见各脚本的详情代码。

---

## 环境要求

- Python ≥ 3.11
- PyTorch ≥ 2.x 且带 CUDA（bfloat16）
- `transformers`、`safetensors`、`PIL`、`rich`、`textual`、fast tokenizer
- bfloat16 推理约需 6–8 GB 显存；训练需要更多（Preview 规模单张消费级 GPU 即可运行）

---

## 评测结果

发布模型的评测成绩（MMLU 等）维护在 [Hugging Face 模型卡](https://huggingface.co/RheoEcho/ETET-1.0-24E-1.8B-A1B-Preview) 上，本仓库不重复维护，以便聚焦架构与训练代码。

---

## 许可证

- 代码：AGPL v3。
- 模型权重：Apache-2.0。
- 基础模型 MiniCPM5-1B-SFT 与 SigLIP-HD 及用到的所有数据库受各自许可证约束。

## 致谢

- 基于 [MiniCPM5-1B-SFT](https://huggingface.co/openbmb/MiniCPM5-1B)（OpenBMB）
- 视觉编码器：[SigLIP-HD](https://huggingface.co/LiheYoung/SigLIP-HD) (LiheYoung)。
- Upcycling 概念受 *Sparse Upcycling*（Komatsuzaki et al., 2023）启发。

# 版权通知
- ETET和RheoEcho字样归RheoEcho所有，未经RheoEcho授权，禁止任何第三方使用。
- 本仓库所有文件和代码均使用 AGPL v3，即便文件内未明确写明，仍具有法律效力，侵权必究！
- 测试图片由mirn提供，版权归他本人所有。
- RheoEcho 2026