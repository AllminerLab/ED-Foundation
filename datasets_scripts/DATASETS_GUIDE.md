# BEiT3-ED Foundation Model - Dataset Guide

This guide explains how to train and run inference with the ED-Foundation Model across the supported datasets.

## Supported Datasets

### 1. Early ED Triage

| Dataset | Modality | Task | Classes | Training |
|--------|------|------|--------|-------------|
| **SYSMH-S-Triage** | Text | Classification | 4 (levels 1-4) | Train |
| **MIMIC-IV-ED-Triage** | Text | Classification | 5 (levels 1-5) | Train |
| **SYSMH-N-Triage** | Text | Classification | 4 (levels 1-4) | Zero-shot |
| **GTCMH-Triage** | Text | Classification | 4 (levels 1-4) | Zero-shot |

### 2. Prognosis Prediction

| Dataset | Modality | Task | Classes | Training |
|--------|------|------|------|-------------|
| **SYSMH-ED-Outcome** | Text | Admission prediction | 2 (Not admitted/Admitted) | Train |
| **MIMIC-IV-ED-AMI** | Text | Mortality prediction | 2 (Survival/Mortality) | Train |
| **MIMIC-IV-ED-Outcome** | Multimodal (text + ECG) | Multi-task (3) | See below | Train |

**MIMIC-IV-ED-Outcome task details**:
- Admission prediction: 2 classes
- Length of stay: 3 classes (Short/Medium/Long)
- Severity scoring: 9 classes (scores 0-8)

### 3. Full-Process ED Decision

| Dataset | Modality | Task | Classes | Training |
|--------|------|------|------|-------------|
| **MIMIC-IV-EXT-MDS-ED** | Multimodal (text + CXR + ECG) | Multi-task (4) | See below | Train |
| **SYSMH-ED-MD** | Text | Multi-task (4) | See below | Train |

**MIMIC-IV-EXT-MDS-ED task details**:
- Mechanical ventilation: 2 classes
- ICU stay: 2 classes
- 7-day mortality: 2 classes
- 28-day mortality: 2 classes

**SYSMH-ED-MD task details**:
- Imaging exam: 2 classes
- Lab tests: 2 classes
- Observation: 2 classes
- Specialist consultation: 2 classes

## Quick Start

### Basic Workflow

```python
from modeling_beit3_ed import BEiT3EDFoundationModel, SimpleEDTokenizer

# 1. Load pretrained model and configure dataset
model = BEiT3EDFoundationModel.from_pretrained(
    'pytorch_model.bin',
    dataset_name='SYSMH-S-Triage',
    task_type='early_triage'
)

# 2. Prepare data
tokenizer = SimpleEDTokenizer()
text = "65-year-old male patient with chest pain for 3 hours"
inputs = tokenizer(text, max_length=512, padding='max_length', return_tensors='pt')

# 3. Inference
predictions, probabilities = model.predict(
    input_ids=inputs['input_ids'],
    attention_mask=inputs['attention_mask']
)

print(f"Predicted class: {predictions}")
print(f"Probability distribution: {probabilities}")
```

## Training Examples

### 1. SYSMH-S-Triage

```bash
python early_triage/train_sysmh_s_triage.py \
    --model_path ../pytorch_model.bin \
    --train_data /path/to/train.csv \
    --val_data /path/to/val.csv \
    --batch_size 32 \
    --num_epochs 10 \
    --learning_rate 1e-3
```

**Data format**:
```csv
text,label
"Patient chief complaint...",1
"Patient chief complaint...",3
```

### 2. MIMIC-IV-ED-Triage

```bash
python early_triage/train_mimic_triage.py \
    --model_path ../pytorch_model.bin \
    --train_data /path/to/train.csv \
    --val_data /path/to/val.csv \
    --batch_size 32
```

### 3. SYSMH-N-Triage

```bash
# Run direct inference with a model trained on SYSMH-S-Triage
python early_triage/infer_sysmh_n_triage.py \
    --model_path checkpoints/sysmh_s_triage/best_model.pt \
    --test_data /path/to/test.csv \
    --output predictions.csv
```

### 4. SYSMH-ED-Outcome

```bash
python prognosis_prediction/train_sysmh_outcome.py \
    --model_path ../pytorch_model.bin \
    --train_data /path/to/train.csv \
    --val_data /path/to/val.csv
```

**Data format**:
```csv
text,label
"Patient data...",0  # Not admitted
"Patient data...",1  # Admitted
```

### 5. MIMIC-IV-ED-AMI

```bash
python prognosis_prediction/train_mimic_ami.py \
    --model_path ../pytorch_model.bin \
    --train_data /path/to/train.csv \
    --val_data /path/to/val.csv
```

### 6. MIMIC-IV-ED-Outcome

```bash
python prognosis_prediction/train_mimic_outcome.py \
    --model_path ../pytorch_model.bin \
    --train_data /path/to/train.csv \
    --val_data /path/to/val.csv \
    --ecg_dir /path/to/ecg_images
```

**Multi-task data format**:
```csv
text,ecg_path,admission,length_of_stay,severity_score
"Patient data...",/path/to/ecg.png,1,2,3
```

### 7. MIMIC-IV-EXT-MDS-ED

```bash
python full_process_decision/train_mimic_mds.py \
    --model_path ../pytorch_model.bin \
    --train_data /path/to/train.csv \
    --val_data /path/to/val.csv \
    --cxr_dir /path/to/chest_xrays \
    --ecg_dir /path/to/ecg_images
```

### 8. SYSMH-ED-MD

```bash
python full_process_decision/train_sysmh_md.py \
    --model_path ../pytorch_model.bin \
    --train_data /path/to/train.csv \
    --val_data /path/to/val.csv
```

**Four-task data format**:
```csv
text,imaging_exam,lab_tests,observation,specialist_consultation
"Patient data...",1,0,1,0
```

## Key Features

### Linear Probing

All training scripts use **linear probing**:
- Encoder parameters are frozen
- Only task heads are trained
- Training is fast
- Overfitting risk is reduced
- Suitable for small datasets

```python
# Encoders are frozen automatically
model = BEiT3EDFoundationModel.from_pretrained(...)
model.freeze_encoders()

# Verify frozen parameters
for name, param in model.named_parameters():
    if 'text_encoder' in name or 'vision_encoder' in name:
        assert not param.requires_grad
    if 'task_head' in name:
        assert param.requires_grad
```

### Zero-shot Inference

SYSMH-N-Triage and GTCMH-Triage do not require additional training:

```python
# 1. Train on SYSMH-S-Triage
model = BEiT3EDFoundationModel.from_pretrained(
    'pytorch_model.bin',
    dataset_name='SYSMH-S-Triage',
    task_type='early_triage'
)
# ... training ...

# 2. Save task head
torch.save(model.task_head.state_dict(), 'sysmh_s_task_head.pt')

# 3. Run zero-shot inference on SYSMH-N-Triage
model_n = BEiT3EDFoundationModel.from_pretrained(
    'pytorch_model.bin',
    dataset_name='SYSMH-N-Triage',
    task_type='early_triage'
)
model_n.task_head.load_state_dict(torch.load('sysmh_s_task_head.pt'))

predictions, probs = model_n.predict(...)
```

## Model Evaluation

### Single-task Evaluation

```python
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Collect predictions and labels
all_preds = []
all_labels = []

for batch in test_loader:
    preds, probs = model.predict(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask']
    )
    all_preds.extend(preds.cpu().numpy())
    all_labels.extend(batch['labels'].cpu().numpy())

# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(classification_report(all_labels, all_preds))
```

### Multi-task Evaluation

```python
for task_name in model.task_head.task_names:
    task_preds = predictions[task_name]
    task_labels = labels[task_name]
    
    acc = accuracy_score(task_labels, task_preds)
    print(f"{task_name}: Accuracy = {acc:.4f}")
```

## Advanced Usage

### Custom Task Head

```python
model = BEiT3EDFoundationModel.from_pretrained(...)

from modeling_beit3_ed import LinearProbingHead
model.task_head = LinearProbingHead(
    input_dim=768,
    num_classes=4,
    hidden_dim=1024
)
```

### Mixed-precision Training

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

## Output Format

Training scripts save:

```
checkpoints/
|-- best_model.pt          # Best model
|-- training_log.json      # Training log
`-- config.json            # Training config
```

Inference scripts generate:

```
predictions.csv            # Predictions
|-- id,prediction,probability
|-- 001,2,0.85
`-- 002,1,0.92
```

## FAQ

### Q: What data format is required?
A: CSV files with text and label columns. Multimodal tasks require image path columns.

### Q: How long does training take?
A: Linear probing usually takes 10-30 minutes depending on dataset size.

### Q: What GPU memory is required?
A: Most tasks can be trained on a single GPU with 8GB or more. Multimodal tasks are better suited to 16GB or more.

### Q: How are Chinese data handled?
A: `SimpleEDTokenizer` supports both Chinese and English text.

### Q: How well does zero-shot inference work?
A: It can approach supervised performance when the target distribution is similar, but target-site evaluation is recommended.

## More Resources

- Model architecture: `modeling_beit3_ed.py`
- Config file: `config.json`
- Training template: `train_template.py`
- Usage examples: `../example_usage.py`

For questions, see the main README or open an Issue.
