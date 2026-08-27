# ETETMoE

[English](README.md) | [簡體中文](README_zh-CN.md) | [繁體中文](README_zh-TW.md)

**ETETMoE** 是 RheoEcho 參考 **Dense-to-MoE Sparse Upcycling** 思想打造的多模態 MoE 架構與流水線，不光致力於讓邊緣設備運行模型，還致力於讓低算力設備成功跑通訓練流水線全過程。利用高效的 MoE 架構讓每一個設備用上高效能多模態大模型。

# 當前版本介紹
## 版本架構
> 1.0

> ETETMoE_LLAMA
## 產出模型系列
1. ETET-1.0-24E
- [ETET-1.0-24E-1.8B-A1B-Preview](https://huggingface.co/RheoEcho/ETET-1.0-24E-1.8B-A1B-Preview) 
## 當前版本介紹
- 它將一個小型稠密因果語言模型（MiniCPM5-1B-SFT）轉換為 **稠密 + 混合專家（Mixture-of-Experts）** 混合架構模型，只需很少的額外訓練，再掛接 SigLIP-HD 視覺編碼器即可支援多模態。
- 經實驗，本架構全部過程只需單卡 NVIDIA Geforce RTX 4060 Laptop 即可完成
---

## 倉庫結構

| 路徑 | 用途 |
| --- | --- |
| `env.py` | 共享環境、常量、`get_logger`、硬體/配置輔助 |
| `download.py` | 下載基礎模型（MiniCPM5-1B-SFT）與 SigLIP-HD 檢查點 |
| `upcycle.py` | **Dense-to-MoE Sparse Upcycling**：構建 ETET-MoE 權重佈局 |
| `train.py` | 階段 2——文字微調自動分化專家職能 |
| `visiontrain.py` | 階段 3——多模態對齊 |
| `benchmark.py` | 評測框架（MMLU 5-shot，文字路徑） |
| `etet_id_datasets.py` | 身分 / 指令微調資料集組裝 |
| `infer_tui.py` | Textual TUI 聊天：支援文字與多模態（VL）ETET 和標準 Llama 模型 |
| `models/` | 本地模型檢查點（每個模型一個子目錄） |
| `output/` | 訓練日誌 / 輸出 |
| `test/` | 測試資源 |


---

## 流水線

```
MiniCPM5-1B-SFT ──upcycle.py──▶ ETETMoE_LLAMA（第 0-15 層稠密 + 第 16-23 層 MoE）
        │
        ├─ train.py ───────▶ ETET-Expert   （在 upcycle 後的模型基礎上做文字 SFT 進行專家分化）
        │
        └─ visiontrain.py ─▶ ETETVL   （掛接 SigLIP-HD + Connector，做 VL SFT）
```

---

## 快速開始

```bash
git clone https://github.com/Rheoecho-Studio/ETETMoE
cd ETETMoE
python env.py

# 1) 取得基礎檢查點
python download.py

# 2) 從稠密基礎模型構建 ETET-MoE 佈局
python upcycle.py

# 3) 文字微調（ETETLM）
python train.py

# 4) 多模態對齊（ETETVL）
python visiontrain.py

# 5) 聊天 / 評測
python infer_tui.py 
python benchmark.py
```

> 每個腳本的完整 CLI 選項（batch size、學習率、epoch、影像尺寸、專家數量、MoE 起止層等）見各腳本的詳情程式碼。

---

## 環境要求

- Python ≥ 3.11
- PyTorch ≥ 2.x 且帶 CUDA（bfloat16）
- `transformers`、`safetensors`、`PIL`、`rich`、`textual`、fast tokenizer
- bfloat16 推論約需 6–8 GB 顯存；訓練需要更多（Preview 規模單張消費級 GPU 即可運行）

---

## 評測結果

發布模型的評測成績（MMLU 等）維護在 [Hugging Face 模型卡](https://huggingface.co/RheoEcho/ETET-1.0-24E-1.8B-A1B-Preview) 上，本倉庫不重複維護，以便聚焦架構與訓練程式碼。

---

## 許可證

- 程式碼：AGPL v3。
- 模型權重：Apache-2.0。
- 基礎模型 MiniCPM5-1B-SFT 與 SigLIP-HD 及用到的所有資料庫受各自許可證約束。

## 致謝

- 基於 [MiniCPM5-1B-SFT](https://huggingface.co/openbmb/MiniCPM5-1B)（OpenBMB）
- 視覺編碼器：[SigLIP-HD](https://huggingface.co/LiheYoung/SigLIP-HD) (LiheYoung)。
- Upcycling 概念受 *Sparse Upcycling*（Komatsuzaki et al., 2023）啟發。

# 版權通知
- ETET 和 RheoEcho 字樣歸 RheoEcho 所有，未經 RheoEcho 授權，禁止任何第三方使用。
- 本倉庫所有檔案和程式碼均使用 AGPL v3，即便檔案內未明確寫明，仍具有法律效力，侵權必究！
- 測試圖片由 mirn 提供，版權歸他本人所有。
- RheoEcho 2026
