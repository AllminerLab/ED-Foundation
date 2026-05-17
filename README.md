# ED-Foundation Model

<div align="center">

**急诊科基础模型 | ED-Foundation Model**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

</div>

<div align="center">
  <a href="#chinese">简体中文</a> | <a href="#english">English</a>
</div>

---

<details open>
<summary id="chinese"><b>简体中文</b></summary>

## 🏥 模型简介

**ED-Foundation Model** 是首个专为急诊科设计的通用多模态医疗基础模型。基于 BEiT3 架构，采用**线性探测 (Linear Probing)** 训练策略，支持多种急诊医疗任务和数据集。

### ✨ 核心特点

- 🎯 **通用性强**: 支持 9 个数据集，覆盖急诊全流程
- 🔀 **多模态**: 同时处理文本、X 光片、心电图 (ECG)
- 🚀 **高效训练**: 线性探测策略，编码器冻结，训练快速
- 🌍 **跨语言**: 支持中英文数据集
- 🔄 **零样本**: 支持跨医院零样本推理
- 📊 **多任务**: 单模型支持多个预测任务

### 🎓 训练策略

**线性探测 (Linear Probing)**：
- 编码器参数**冻结**（~220M 参数）
- 只训练任务头（~1-5M 参数）
- 训练时间短（10-30 分钟）
- 避免过拟合，适合小样本

## 📊 支持的数据集和任务

### 任务类型一：早期急诊分诊 (Early ED Triage)

针对急诊患者的快速分级分类任务。

| 数据集 | 语言 | 类别 | 训练方式 | 说明 |
|--------|------|------|----------|------|
| **SYSMH-S-Triage** | 中文 | 4 分类 (1-4 级) | ✅ 训练 | 中山大学孙逸仙纪念医院南院区急诊分诊 |
| **MIMIC-IV-ED-Triage** | 英文 | 5 分类 (1-5 级) | ✅ 训练 | MIMIC-IV 急诊分诊 |
| **SYSMH-N-Triage** | 中文 | 4 分类 (1-4 级) | 🔄 零样本 | 中山大学孙逸仙纪念医院北院区（零样本） |
| **GTCMH-Triage** | 中文 | 4 分类 (1-4 级) | 🔄 零样本 | 广东省中医院（零样本） |

**模态**: 纯文本（病历、生命体征）

### 任务类型二：预后预测 (Prognosis Prediction)

预测患者住院、死亡率等预后指标。

| 数据集 | 语言 | 任务 | 类别 | 模态 |
|--------|------|------|------|------|
| **SYSMH-ED-Outcome** | 中文 | 住院预测 | 2 分类 | 文本 |
| **MIMIC-IV-ED-AMI** | 英文 | 院内死亡预测 | 2 分类 | 文本 |
| **MIMIC-IV-ED-Outcome** | 英文 | 多任务 (3) | 见下 | 文本 + ECG |

**MIMIC-IV-ED-Outcome 多任务详情**:
1. **Admission Prediction** (住院预测): 2 分类 - Not admitted / Admitted
2. **Length of Stay** (停留时间): 3 分类 - Short / Medium / Long
3. **Severity Scoring** (严重评分): 9 分类 - Score 0-8

### 任务类型三：全流程急诊决策 (Full-Process ED Decision)

支持急诊全流程的医疗决策支持。

| 数据集 | 语言 | 任务数 | 模态 |
|--------|------|--------|------|
| **MIMIC-IV-EXT-MDS-ED** | 英文 | 4 任务 | 文本 + CXR + ECG |
| **SYSMH-ED-MD** | 中文 | 4 任务 | 文本 |

**MIMIC-IV-EXT-MDS-ED 任务**:
1. **Mechanical Ventilation** (机械通气): 需要/不需要
2. **ICU Stay** (ICU 转诊): 需要/不需要
3. **7-day Mortality** (7 天死亡率): 存活/死亡
4. **28-day Mortality** (28 天死亡率): 存活/死亡

**SYSMH-ED-MD 任务**:
1. **Imaging Exam** (影像检查): 需要/不需要
2. **Lab Tests** (实验室检查): 需要/不需要
3. **Observation** (留观): 需要/不需要
4. **Specialist Consultation** (专科会诊): 需要/不需要

## 🚀 快速开始

### 模型权重获取

由于预训练权重文件较大（约 9.8GB），仓库中不直接包含 `pytorch_model.bin`。如需获取完整权重，请在审稿流程中联系通讯作者。

### 安装依赖

```bash
pip install torch torchvision timm transformers pillow
```

### 基本使用

```python
from modeling_beit3_ed import BEiT3EDFoundationModel, SimpleEDTokenizer

# 加载模型并配置数据集
model = BEiT3EDFoundationModel.from_pretrained(
    'pytorch_model.bin',
    dataset_name='SYSMH-S-Triage',  # 指定数据集
    task_type='early_triage'        # 指定任务类型
)
```

## 🔧 训练指南

### Methods 复现工具

仓库提供 `methods_protocol.json` 和 `methods_reproducibility.py`，用于对齐论文 Methods 中的数据集 schema、输入构造、缺失值指示、患者级划分、7 次运行指标汇总、校准误差和分诊资源模拟。所有中间文件默认生成到同一个目录 `artifacts/methods/`，便于审稿和上传仓库时保持文件结构清晰。

```bash
python methods_reproducibility.py \
  --artifact-dir artifacts/methods \
  prepare-dataset \
  --dataset-name SYSMH-S-Triage \
  --input-csv /path/to/sysmh_s_triage.csv
```

详细对应关系见 [METHODS_COMPLIANCE.md](METHODS_COMPLIANCE.md)。

### 线性探测训练

```bash
cd datasets_scripts

# 训练 SYSMH-S-Triage 分诊模型
python early_triage/train_sysmh_s_triage.py \
    --model_path ../pytorch_model.bin \
    --train_data /path/to/train.csv \
    --val_data /path/to/val.csv \
    --batch_size 32 \
    --num_epochs 10 \
    --learning_rate 1e-3
```

### 零样本推理

```bash
# 使用 SYSMH-S-Triage 训练的模型在 SYSMH-N-Triage 上零样本推理
python early_triage/infer_sysmh_n_triage.py \
    --model_path checkpoints/sysmh_s_triage/best_model.pt \
    --test_data /path/to/test.csv \
    --output predictions.csv
```

详细训练指南请参考：[datasets_scripts/DATASETS_GUIDE.md](datasets_scripts/DATASETS_GUIDE.md)

## 📈 模型架构

```
输入层
  ├─ 文本输入 → 文本编码器 (12-layer Transformer, 135M 参数) [冻结]
  └─ 图像输入 → 视觉编码器 (BEiT Base, 86M 参数) [冻结]
           ↓
     多模态融合层 (768 维) [冻结]
           ↓
     线性探测头 (1-5M 参数) [可训练]
           ↓
        预测输出
```

### 参数规模

| 组件 | 参数量 | 训练状态 |
|------|--------|----------|
| 文本编码器 | ~135M | ❄️ 冻结 |
| 视觉编码器 | ~86M | ❄️ 冻结 |
| 融合层 | ~1M | ❄️ 冻结 |
| 任务头 | ~1-5M | ✅ 可训练 |
| **总计** | **~220M** | **线性探测** |

## 📦 文件结构

```
huggingface_model/
├── pytorch_model.bin           # 预训练权重 (9.8GB)
├── config.json                 # 模型配置
├── modeling_beit3_ed.py        # 模型定义
├── tokenizer_config.json       # Tokenizer 配置
├── README.md                   # 本文件
├── requirements.txt            # 依赖列表
├── LICENSE                     # Apache 2.0 许可证
├── example_usage.py            # 使用示例
├── upload_to_huggingface.py    # 上传脚本
└── datasets_scripts/           # 数据集脚本
    ├── DATASETS_GUIDE.md       # 数据集详细指南
    ├── train_template.py       # 训练模板
    ├── early_triage/           # 早期分诊脚本
    ├── prognosis_prediction/   # 预后预测脚本
    └── MDs/                    # 全流程决策脚本
```

## ⚠️ 重要说明

### 适用范围
- ✅ 研究和开发
- ✅ 模型评估和比较
- ✅ 教育和培训
- ❌ **不能直接用于临床决策**

### 限制
1. **需要临床验证**: 所有预测需结合临床判断
2. **数据分布**: 模型在特定数据集上训练，可能存在分布偏差
3. **零样本性能**: 跨医院零样本推理效果可能有差异
4. **伦理考虑**: 注意患者隐私保护，符合医疗伦理规范

## 📄 许可证

本模型遵循 [Apache 2.0 License](LICENSE)

## 🤝 贡献

欢迎贡献！请提交 Issue 或 Pull Request。

## 🙏 致谢

- **数据集**: MIMIC-IV-ED, SYSMH, GTCMH
- **基础模型**: [BEiT](https://github.com/microsoft/unilm/tree/master/beit)
- **库**: [timm](https://github.com/huggingface/pytorch-image-models), [transformers](https://github.com/huggingface/transformers)

</details>

<details>
<summary id="english"><b>English</b></summary>

## 🏥 Model Overview

**ED-Foundation Model** is the first general-purpose multimodal medical foundation model designed specifically for the Emergency Department (ED). It is built on the BEiT3 architecture and trained with **Linear Probing**, supporting a wide range of ED tasks and datasets.

### ✨ Key Features

- 🎯 **General-purpose**: supports 9 datasets spanning the full ED workflow
- 🔀 **Multimodal**: jointly handles text, X-rays, and ECG
- 🚀 **Efficient training**: linear probing with frozen encoders for fast training
- 🌍 **Cross-lingual**: supports Chinese and English datasets
- 🔄 **Zero-shot**: enables cross-hospital zero-shot inference
- 📊 **Multi-task**: a single model supports multiple prediction tasks

### 🎓 Training Strategy

**Linear Probing**:
- Encoders are **frozen** (~220M parameters)
- Only task heads are trained (~1-5M parameters)
- Short training time (10-30 minutes)
- Reduced overfitting, good for small datasets

## 📊 Supported Datasets & Tasks

### Task Type 1: Early ED Triage

Fast acuity-level classification for ED patients.

| Dataset | Language | Classes | Training | Notes |
|--------|------|------|----------|------|
| **SYSMH-S-Triage** | Chinese | 4 classes (Level 1-4) | ✅ Train | ED triage at SYSU Sun Yat-sen Memorial Hospital (South Campus) |
| **MIMIC-IV-ED-Triage** | English | 5 classes (Level 1-5) | ✅ Train | MIMIC-IV ED triage |
| **SYSMH-N-Triage** | Chinese | 4 classes (Level 1-4) | 🔄 Zero-shot | SYSU SMH (North Campus, zero-shot) |
| **GTCMH-Triage** | Chinese | 4 classes (Level 1-4) | 🔄 Zero-shot | Guangdong Provincial TCM Hospital (zero-shot) |

**Modality**: text only (notes, vitals)

### Task Type 2: Prognosis Prediction

Predict outcomes such as admission and mortality.

| Dataset | Language | Task | Classes | Modality |
|--------|------|------|------|------|
| **SYSMH-ED-Outcome** | Chinese | Admission | 2 classes | Text |
| **MIMIC-IV-ED-AMI** | English | In-hospital Mortality | 2 classes | Text |
| **MIMIC-IV-ED-Outcome** | English | Multi-task (3) | See below | Text + ECG |

**MIMIC-IV-ED-Outcome Multi-task Details**:
1. **Admission Prediction**: 2 classes - Not admitted / Admitted
2. **Length of Stay**: 3 classes - Short / Medium / Long
3. **Severity Scoring**: 9 classes - Score 0-8

### Task Type 3: Full-Process ED Decision

Decision support across the full ED workflow.

| Dataset | Language | #Tasks | Modality |
|--------|------|--------|------|
| **MIMIC-IV-EXT-MDS-ED** | English | 4 tasks | Text + CXR + ECG |
| **SYSMH-ED-MD** | Chinese | 4 tasks | Text |

**MIMIC-IV-EXT-MDS-ED Tasks**:
1. **Mechanical Ventilation**: Need / No need
2. **ICU Stay**: Need / No need
3. **7-day Mortality**: Survived / Died
4. **28-day Mortality**: Survived / Died

**SYSMH-ED-MD Tasks**:
1. **Imaging Exam**: Need / No need
2. **Lab Tests**: Need / No need
3. **Observation**: Need / No need
4. **Specialist Consultation**: Need / No need

## 🚀 Quick Start

### Model Weights

The pretrained weight file (`pytorch_model.bin`) is large (about 9.8GB) and is not included directly in this repository. The full weights are available upon request.

### Install Dependencies

```bash
pip install torch torchvision timm transformers pillow
```

### Basic Usage

```python
from modeling_beit3_ed import BEiT3EDFoundationModel, SimpleEDTokenizer

# Load model and configure dataset
model = BEiT3EDFoundationModel.from_pretrained(
    'pytorch_model.bin',
    dataset_name='SYSMH-S-Triage',  # Dataset
    task_type='early_triage'        # Task type
)
```

## 🔧 Training Guide

### Methods Reproducibility Utilities

The repository includes `methods_protocol.json` and `methods_reproducibility.py` to align the code release with the manuscript Methods, including dataset schemas, input construction, missingness indicators, patient-level splitting, 7-run metric aggregation, calibration error, and triage resource simulation. All intermediate files are generated under one directory by default: `artifacts/methods/`.

```bash
python methods_reproducibility.py \
  --artifact-dir artifacts/methods \
  prepare-dataset \
  --dataset-name SYSMH-S-Triage \
  --input-csv /path/to/sysmh_s_triage.csv
```

See [METHODS_COMPLIANCE.md](METHODS_COMPLIANCE.md) for the Methods-to-code mapping.

### Linear Probing Training

```bash
cd datasets_scripts

# Train SYSMH-S-Triage model
python early_triage/train_sysmh_s_triage.py \
    --model_path ../pytorch_model.bin \
    --train_data /path/to/train.csv \
    --val_data /path/to/val.csv \
    --batch_size 32 \
    --num_epochs 10 \
    --learning_rate 1e-3
```

### Zero-shot Inference

```bash
# Zero-shot on SYSMH-N-Triage using a model trained on SYSMH-S-Triage
python early_triage/infer_sysmh_n_triage.py \
    --model_path checkpoints/sysmh_s_triage/best_model.pt \
    --test_data /path/to/test.csv \
    --output predictions.csv
```

For detailed training guidance, see: [datasets_scripts/DATASETS_GUIDE.md](datasets_scripts/DATASETS_GUIDE.md)

## 📈 Model Architecture

```
Inputs
  ├─ Text input → Text Encoder (12-layer Transformer, 135M params) [Frozen]
  └─ Image input → Vision Encoder (BEiT Base, 86M params) [Frozen]
           ↓
     Multimodal Fusion (768 dim) [Frozen]
           ↓
     Linear Probing Head (1-5M params) [Trainable]
           ↓
        Outputs
```

### Parameter Scale

| Component | Parameters | Training |
|------|--------|----------|
| Text Encoder | ~135M | ❄️ Frozen |
| Vision Encoder | ~86M | ❄️ Frozen |
| Fusion Layer | ~1M | ❄️ Frozen |
| Task Heads | ~1-5M | ✅ Trainable |
| **Total** | **~220M** | **Linear Probing** |

## 📦 File Structure

```
huggingface_model/
├── pytorch_model.bin           # Pretrained weights (9.8GB)
├── config.json                 # Model config
├── modeling_beit3_ed.py        # Model definition
├── tokenizer_config.json       # Tokenizer config
├── README.md                   # This file
├── requirements.txt            # Dependencies
├── LICENSE                     # Apache 2.0 License
├── example_usage.py            # Example usage
├── upload_to_huggingface.py    # Upload script
└── datasets_scripts/           # Dataset scripts
    ├── DATASETS_GUIDE.md       # Dataset guide
    ├── train_template.py       # Training template
    ├── early_triage/           # Early triage scripts
    ├── prognosis_prediction/   # Prognosis prediction scripts
    └── MDs/                    # Full-process decision scripts
```

## ⚠️ Important Notes

### Scope
- ✅ Research and development
- ✅ Model evaluation and benchmarking
- ✅ Education and training
- ❌ **Not for direct clinical decision-making**

### Limitations
1. **Clinical validation required**: predictions must be interpreted with clinical judgment
2. **Data shift**: trained on specific datasets; distribution bias may exist
3. **Zero-shot variability**: performance may vary across hospitals
4. **Ethics**: protect patient privacy and follow medical ethics

## 📄 License

This model is released under the [Apache 2.0 License](LICENSE)

## 🤝 Contributing

Contributions are welcome! Please open an Issue or Pull Request.

## 🙏 Acknowledgements

- **Datasets**: MIMIC-IV-ED, SYSMH, GTCMH
- **Base Model**: [BEiT](https://github.com/microsoft/unilm/tree/master/beit)
- **Libraries**: [timm](https://github.com/huggingface/pytorch-image-models), [transformers](https://github.com/huggingface/transformers)

</details>
