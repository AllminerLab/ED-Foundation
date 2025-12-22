# BEiT3-ED Foundation Model

<div align="center">

**急诊科基础模型 | ED-Foundation Model**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

</div>

---

<a name="chinese"></a>

## 🏥 模型简介

**ED-Foundation Model** 是首个专为急诊科设计的通用多模态医疗基础模型。基于BEiT3架构，采用**线性探测(Linear Probing)**训练策略，支持多种急诊医疗任务和数据集。

### ✨ 核心特点

- 🎯 **通用性强**: 支持9个数据集，覆盖急诊全流程
- 🔀 **多模态**: 同时处理文本、X光片、心电图(ECG)
- 🚀 **高效训练**: 线性探测策略，编码器冻结，训练快速
- 🌍 **跨语言**: 支持中英文数据集
- 🔄 **零样本**: 支持跨医院零样本推理
- 📊 **多任务**: 单模型支持多个预测任务

### 🎓 训练策略

**线性探测 (Linear Probing)**：
- 编码器参数**冻结**（~220M参数）
- 只训练任务头（~1-5M参数）
- 训练时间短（10-30分钟）
- 避免过拟合，适合小样本

## 📊 支持的数据集和任务

### 任务类型一：早期急诊分诊 (Early ED Triage)

针对急诊患者的快速分级分类任务。

| 数据集 | 语言 | 类别 | 训练方式 | 说明 |
|--------|------|------|----------|------|
| **SYSMH-S-Triage** | 中文 | 4分类 (1-4级) | ✅ 训练 | 中山大学孙逸仙纪念医院南院区急诊分诊 |
| **MIMIC-IV-ED-Triage** | 英文 | 5分类 (1-5级) | ✅ 训练 | MIMIC-IV急诊分诊 |
| **SYSMH-N-Triage** | 中文 | 4分类 (1-4级) | 🔄 零样本 | 中山大学孙逸仙纪念医院北院区（零样本） |
| **GTCMH-Triage** | 中文 | 4分类 (1-4级) | 🔄 零样本 | 广东省中医院（零样本） |

**模态**: 纯文本（病历、生命体征）

### 任务类型二：预后预测 (Prognosis Prediction)

预测患者住院、死亡率等预后指标。

| 数据集 | 语言 | 任务 | 类别 | 模态 |
|--------|------|------|------|------|
| **SYSMH-ED-Outcome** | 中文 | 住院预测 | 2分类 | 文本 |
| **MIMIC-IV-ED-AMI** | 英文 | 院内死亡预测 | 2分类 | 文本 |
| **MIMIC-IV-ED-Outcome** | 英文 | 多任务(3) | 见下 | 文本+ECG |

**MIMIC-IV-ED-Outcome 多任务详情**：
1. **Admission Prediction** (住院预测): 2分类 - Not admitted / Admitted
2. **Length of Stay** (停留时间): 3分类 - Short / Medium / Long
3. **Severity Scoring** (严重评分): 9分类 - Score 0-8

### 任务类型三：全流程急诊决策 (Full-Process ED Decision)

支持急诊全流程的医疗决策支持。

| 数据集 | 语言 | 任务数 | 模态 |
|--------|------|--------|------|
| **MIMIC-IV-EXT-MDS-ED** | 英文 | 4任务 | 文本+CXR+ECG |
| **SYSMH-ED-MD** | 中文 | 4任务 | 文本 |

**MIMIC-IV-EXT-MDS-ED 任务**：
1. **Mechanical Ventilation** (机械通气): 需要/不需要
2. **ICU Stay** (ICU转诊): 需要/不需要
3. **7-day Mortality** (7天死亡率): 存活/死亡
4. **28-day Mortality** (28天死亡率): 存活/死亡

**SYSMH-ED-MD 任务**：
1. **Imaging Exam** (影像检查): 需要/不需要
2. **Lab Tests** (实验室检查): 需要/不需要
3. **Observation** (留观): 需要/不需要
4. **Specialist Consultation** (专科会诊): 需要/不需要

## 🚀 快速开始

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
    task_type='early_triage'          # 指定任务类型
)

## 🔧 训练指南

### 线性探测训练

```bash
cd datasets_scripts

# 训练SYSMH-S-Triage分诊模型
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
# 使用SYSMH-S-Triage训练的模型在SYSMH-N-Triage上零样本推理
python early_triage/infer_sysmh_n_triage.py \
    --model_path checkpoints/sysmh_s_triage/best_model.pt \
    --test_data /path/to/test.csv \
    --output predictions.csv
```


详细训练指南请参考：[datasets_scripts/DATASETS_GUIDE.md](datasets_scripts/DATASETS_GUIDE.md)

## 📈 模型架构

```
输入层
  ├─ 文本输入 → 文本编码器 (12-layer Transformer, 135M参数) [冻结]
  └─ 图像输入 → 视觉编码器 (BEiT Base, 86M参数) [冻结]
           ↓
     多模态融合层 (768维) [冻结]
           ↓
     线性探测头 (1-5M参数) [可训练]
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
├── tokenizer_config.json       # Tokenizer配置
├── README.md                   # 本文件
├── requirements.txt            # 依赖列表
├── LICENSE                     # Apache 2.0许可证
├── example_usage.py            # 使用示例
├── upload_to_huggingface.py    # 上传脚本
└── datasets_scripts/           # 数据集脚本
    ├── DATASETS_GUIDE.md       # 数据集详细指南
    ├── train_template.py       # 训练模板
    ├── early_triage/           # 早期分诊脚本
    ├── prognosis_prediction/   # 预后预测脚本
    └── MDs/  # 全流程决策脚本
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

欢迎贡献！请提交Issue或Pull Request。
## 🙏 致谢

- **数据集**: MIMIC-IV-ED, SYSMH, GTCMH
- **基础模型**: [BEiT](https://github.com/microsoft/unilm/tree/master/beit)
- **库**: [timm](https://github.com/huggingface/pytorch-image-models), [transformers](https://github.com/huggingface/transformers)
