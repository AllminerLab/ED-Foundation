# BEiT3-ED Foundation Model - 数据集使用指南

本指南详细说明如何使用ED Foundation Model在各个数据集上进行训练和推理。

## 📊 支持的数据集总览

### 1. 早期急诊分诊 (Early ED Triage)

| 数据集 | 模态 | 任务 | 类别数 | 训练/零样本 |
|--------|------|------|--------|-------------|
| **SYSMH-S-Triage** | 文本 | 分类 | 4 (1-4级) | ✅ 训练 |
| **MIMIC-IV-ED-Triage** | 文本 | 分类 | 5 (1-5级) | ✅ 训练 |
| **SYSMH-N-Triage** | 文本 | 分类 | 4 (1-4级) | 🔄 零样本 |
| **GTCMH-Triage** | 文本 | 分类 | 4 (1-4级) | 🔄 零样本 |

### 2. 预后预测 (Prognosis Prediction)

| 数据集 | 模态 | 任务 | 类别 | 训练/零样本 |
|--------|------|------|------|-------------|
| **SYSMH-ED-Outcome** | 文本 | 住院预测 | 2 (Not admitted/Admitted) | ✅ 训练 |
| **MIMIC-IV-ED-AMI** | 文本 | 死亡预测 | 2 (Survival/Mortality) | ✅ 训练 |
| **MIMIC-IV-ED-Outcome** | 多模态 (文本+ECG) | 多任务(3) | 见下 | ✅ 训练 |

**MIMIC-IV-ED-Outcome 任务详情**：
- 住院预测 (Admission): 2分类
- 停留时间 (Length of stay): 3分类 (Short/Medium/Long)
- 严重评分 (Severity): 9分类 (0-8分)

### 3. 全流程急诊决策 (Full-Process ED Decision)

| 数据集 | 模态 | 任务 | 类别 | 训练/零样本 |
|--------|------|------|------|-------------|
| **MIMIC-IV-EXT-MDS-ED** | 多模态 (文本+CXR+ECG) | 多任务(4) | 见下 | ✅ 训练 |
| **SYSMH-ED-MD** | 文本 | 多任务(4) | 见下 | ✅ 训练 |

**MIMIC-IV-EXT-MDS-ED 任务详情**：
- Mechanical Ventilation (机械通气): 2分类
- ICU Stay (ICU转诊): 2分类
- 7-day Mortality (7天死亡率): 2分类
- 28-day Mortality (28天死亡率): 2分类

**SYSMH-ED-MD 任务详情**：
- Imaging Exam (影像检查需求): 2分类
- Lab Tests (实验室检查需求): 2分类
- Observation (观察需求): 2分类
- Specialist Consultation (专科会诊需求): 2分类

## 🚀 快速开始

### 基本使用流程

```python
from modeling_beit3_ed import BEiT3EDFoundationModel, SimpleEDTokenizer

# 1. 加载预训练模型并配置数据集
model = BEiT3EDFoundationModel.from_pretrained(
    'pytorch_model.bin',
    dataset_name='SYSMH-S-Triage',
    task_type='early_triage'
)

# 2. 准备数据
tokenizer = SimpleEDTokenizer()
text = "患者年龄65岁，男性，主诉胸痛3小时"
inputs = tokenizer(text, max_length=512, padding='max_length', return_tensors='pt')

# 3. 推理
predictions, probabilities = model.predict(
    input_ids=inputs['input_ids'],
    attention_mask=inputs['attention_mask']
)

print(f"预测类别: {predictions}")
print(f"概率分布: {probabilities}")
```

## 📝 各数据集训练示例

### 1. SYSMH-S-Triage (四分类分诊)

```bash
python early_triage/train_sysmh_s_triage.py \
    --model_path ../pytorch_model.bin \
    --train_data /path/to/train.csv \
    --val_data /path/to/val.csv \
    --batch_size 32 \
    --num_epochs 10 \
    --learning_rate 1e-3
```

**数据格式**：
```csv
text,label
"患者主诉...",1
"患者主诉...",3
```

### 2. MIMIC-IV-ED-Triage (五分类分诊)

```bash
python early_triage/train_mimic_triage.py \
    --model_path ../pytorch_model.bin \
    --train_data /path/to/train.csv \
    --val_data /path/to/val.csv \
    --batch_size 32
```

### 3. SYSMH-N-Triage (零样本推理)

```bash
# 使用SYSMH-S-Triage训练的模型直接推理
python early_triage/infer_sysmh_n_triage.py \
    --model_path checkpoints/sysmh_s_triage/best_model.pt \
    --test_data /path/to/test.csv \
    --output predictions.csv
```

### 4. SYSMH-ED-Outcome (住院预测)

```bash
python prognosis_prediction/train_sysmh_outcome.py \
    --model_path ../pytorch_model.bin \
    --train_data /path/to/train.csv \
    --val_data /path/to/val.csv
```

**数据格式**：
```csv
text,label
"患者数据...",0  # Not admitted
"患者数据...",1  # Admitted
```

### 5. MIMIC-IV-ED-AMI (死亡预测)

```bash
python prognosis_prediction/train_mimic_ami.py \
    --model_path ../pytorch_model.bin \
    --train_data /path/to/train.csv \
    --val_data /path/to/val.csv
```

### 6. MIMIC-IV-ED-Outcome (多模态多任务)

```bash
python prognosis_prediction/train_mimic_outcome.py \
    --model_path ../pytorch_model.bin \
    --train_data /path/to/train.csv \
    --val_data /path/to/val.csv \
    --ecg_dir /path/to/ecg_images
```

**数据格式**（多任务）：
```csv
text,ecg_path,admission,length_of_stay,severity_score
"患者数据...",/path/to/ecg.png,1,2,3
```

### 7. MIMIC-IV-EXT-MDS-ED (多模态多任务决策)

```bash
python full_process_decision/train_mimic_mds.py \
    --model_path ../pytorch_model.bin \
    --train_data /path/to/train.csv \
    --val_data /path/to/val.csv \
    --cxr_dir /path/to/chest_xrays \
    --ecg_dir /path/to/ecg_images
```

### 8. SYSMH-ED-MD (医疗决策多任务)

```bash
python full_process_decision/train_sysmh_md.py \
    --model_path ../pytorch_model.bin \
    --train_data /path/to/train.csv \
    --val_data /path/to/val.csv
```

**数据格式**（四任务）：
```csv
text,imaging_exam,lab_tests,observation,specialist_consultation
"患者数据...",1,0,1,0
```

## 💡 关键特性

### 线性探测 (Linear Probing)

所有训练都采用**线性探测**策略：
- ✅ 编码器参数冻结
- ✅ 只训练任务头
- ✅ 训练速度快
- ✅ 避免过拟合
- ✅ 适合小样本

```python
# 编码器自动冻结
model = BEiT3EDFoundationModel.from_pretrained(...)
model.freeze_encoders()  # 已自动执行

# 验证参数冻结状态
for name, param in model.named_parameters():
    if 'text_encoder' in name or 'vision_encoder' in name:
        assert not param.requires_grad
    if 'task_head' in name:
        assert param.requires_grad
```

### 零样本推理

SYSMH-N-Triage 和 GTCMH-Triage 不需要训练：

```python
# 1. 在SYSMH-S-Triage上训练
model = BEiT3EDFoundationModel.from_pretrained(
    'pytorch_model.bin',
    dataset_name='SYSMH-S-Triage',
    task_type='early_triage'
)
# ... 训练 ...

# 2. 保存任务头
torch.save(model.task_head.state_dict(), 'sysmh_s_task_head.pt')

# 3. 在SYSMH-N-Triage上零样本推理
model_n = BEiT3EDFoundationModel.from_pretrained(
    'pytorch_model.bin',
    dataset_name='SYSMH-N-Triage',  # 配置相同
    task_type='early_triage'
)
model_n.task_head.load_state_dict(torch.load('sysmh_s_task_head.pt'))

# 直接推理，无需训练
predictions, probs = model_n.predict(...)
```

## 📊 模型评估

### 单任务评估

```python
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 收集预测和真实标签
all_preds = []
all_labels = []

for batch in test_loader:
    preds, probs = model.predict(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask']
    )
    all_preds.extend(preds.cpu().numpy())
    all_labels.extend(batch['labels'].cpu().numpy())

# 计算指标
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(classification_report(all_labels, all_preds))
```

### 多任务评估

```python
# 对于多任务模型
for task_name in model.task_head.task_names:
    task_preds = predictions[task_name]
    task_labels = labels[task_name]
    
    acc = accuracy_score(task_labels, task_preds)
    print(f"{task_name}: Accuracy = {acc:.4f}")
```

## 🔧 高级用法

### 自定义任务头

```python
# 如果需要更深的任务头
model = BEiT3EDFoundationModel.from_pretrained(...)

# 替换任务头
from modeling_beit3_ed import LinearProbingHead
model.task_head = LinearProbingHead(
    input_dim=768,
    num_classes=4,
    hidden_dim=1024  # 更大的隐藏层
)
```

### 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    with autocast():
        outputs = model(...)
        loss = outputs['loss']
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## 📁 输出格式

训练脚本会保存：

```
checkpoints/
├── best_model.pt          # 最佳模型
├── training_log.json      # 训练日志
└── config.json            # 训练配置
```

推理脚本会生成：

```
predictions.csv            # 预测结果
├── id,prediction,probability
├── 001,2,0.85
└── 002,1,0.92
```

## 🆘 常见问题

### Q: 数据格式要求？
A: CSV格式，包含文本列和标签列。多模态需要图像路径列。

### Q: 训练时间？
A: 线性探测通常10-30分钟（取决于数据量）。

### Q: 显存需求？
A: 单GPU 8GB+ 可训练大部分任务。多模态任务建议16GB+。

### Q: 如何处理中文数据？
A: SimpleEDTokenizer自动支持中英文。

### Q: 零样本效果如何？
A: 在相似分布数据上效果接近有监督。建议在目标数据上评估。

## 📚 更多资源

- 模型架构: `modeling_beit3_ed.py`
- 配置文件: `config.json`
- 训练模板: `train_template.py`
- 使用示例: `../example_usage.py`

---

如有问题，请参考主README或提交Issue。

