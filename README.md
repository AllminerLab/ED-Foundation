# ED-Foundation Model

<div align="center">

**ED-Foundation Model**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

</div>

## Model Overview

**ED-Foundation Model** is a general-purpose multimodal medical foundation model designed specifically for the Emergency Department (ED). It is built on the BEiT3 architecture and trained with **linear probing**, supporting a wide range of ED tasks and datasets.

### Key Features

- **General-purpose**: supports 9 datasets spanning the full ED workflow
- **Multimodal**: jointly handles text, X-rays, and ECG
- **Efficient training**: linear probing with frozen encoders for fast training
- **Cross-lingual**: supports Chinese and English datasets
- **Zero-shot**: enables cross-hospital zero-shot inference
- **Multi-task**: a single model supports multiple prediction tasks

### Training Strategy

**Linear probing**:
- Encoders are **frozen** (~220M parameters)
- Only task heads are trained (~1-5M parameters)
- Short training time (10-30 minutes)
- Reduced overfitting, suitable for small datasets

## Supported Datasets and Tasks

### Task Type 1: Early ED Triage

Fast acuity-level classification for ED patients.

| Dataset | Language | Classes | Training | Notes |
|--------|------|------|----------|------|
| **SYSMH-S-Triage** | Chinese | 4 classes (Level 1-4) | Train | ED triage at SYSU Sun Yat-sen Memorial Hospital (South Campus) |
| **MIMIC-IV-ED-Triage** | English | 5 classes (Level 1-5) | Train | MIMIC-IV ED triage |
| **SYSMH-N-Triage** | Chinese | 4 classes (Level 1-4) | Zero-shot | SYSU SMH (North Campus, zero-shot) |
| **GTCMH-Triage** | Chinese | 4 classes (Level 1-4) | Zero-shot | Guangdong Provincial TCM Hospital (zero-shot) |

**Modality**: text only (notes, vitals)

### Task Type 2: Prognosis Prediction

Predict outcomes such as admission and mortality.

| Dataset | Language | Task | Classes | Modality |
|--------|------|------|------|------|
| **SYSMH-ED-Outcome** | Chinese | Admission | 2 classes | Text |
| **MIMIC-IV-ED-AMI** | English | In-hospital mortality | 2 classes | Text |
| **MIMIC-IV-ED-Outcome** | English | Multi-task (3) | See below | Text + ECG |

**MIMIC-IV-ED-Outcome multi-task details**:
1. **Admission Prediction**: 2 classes - Not admitted / Admitted
2. **Length of Stay**: 3 classes - Short / Medium / Long
3. **Severity Scoring**: 9 classes - Score 0-8

### Task Type 3: Full-Process ED Decision

Decision support across the full ED workflow.

| Dataset | Language | Tasks | Modality |
|--------|------|--------|------|
| **MIMIC-IV-EXT-MDS-ED** | English | 4 tasks | Text + CXR + ECG |
| **SYSMH-ED-MD** | Chinese | 4 tasks | Text |

**MIMIC-IV-EXT-MDS-ED tasks**:
1. **Mechanical Ventilation**: Need / No need
2. **ICU Stay**: Need / No need
3. **7-day Mortality**: Survived / Died
4. **28-day Mortality**: Survived / Died

**SYSMH-ED-MD tasks**:
1. **Imaging Exam**: Need / No need
2. **Lab Tests**: Need / No need
3. **Observation**: Need / No need
4. **Specialist Consultation**: Need / No need

## Quick Start

### Model Weights

The pretrained weight file (`pytorch_model.bin`) is large (about 9.8GB) and is not included directly in this repository. The pretrained weights and downstream fine-tuned weights will be released after publication of the manuscript.

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

## Training Guide

### Methods Reproducibility Utilities

The repository includes `methods_protocol.json` and `methods_reproducibility.py` include dataset schemas, input construction, missingness indicators, patient-level splitting, 7-run metric aggregation, calibration error, and triage resource simulation. All intermediate files are generated under one directory by default: `artifacts/methods/`.

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

For detailed training guidance, see [datasets_scripts/DATASETS_GUIDE.md](datasets_scripts/DATASETS_GUIDE.md).

## Model Architecture

```
Inputs
  |-- Text input -> Text Encoder (12-layer Transformer, 135M params) [Frozen]
  `-- Image input -> Vision Encoder (BEiT Base, 86M params) [Frozen]
           |
     Multimodal Fusion (768 dim) [Frozen]
           |
     Linear Probing Head (1-5M params) [Trainable]
           |
        Outputs
```

### Parameter Scale

| Component | Parameters | Training |
|------|--------|----------|
| Text Encoder | ~135M | Frozen |
| Vision Encoder | ~86M | Frozen |
| Fusion Layer | ~1M | Frozen |
| Task Heads | ~1-5M | Trainable |
| **Total** | **~220M** | **Linear probing** |

## Data Availability

The SYSMH-ED data used in pretraining are not publicly available and require access through a data use agreement with SYSMH. The original data from the MIMIC-IV-ED database and its associated modules, provided by the Beth Israel Deaconess Medical Center, are available via PhysioNet (access requires certification for researchers, [https://physionet.org/content/mimic-iv-ed/2.2/](https://physionet.org/content/mimic-iv-ed/2.2/)). Access to the Quilt-1M pathology repository requires an official application process. Applicants can visit its official website ([https://quilt1m.github.io/](https://quilt1m.github.io/)) and complete the provided form. Access to the RATIC dataset requires a formal application via email. This dataset is managed by the Radiological Society of North America (RSNA). Applicants need to visit its official page ([https://mira.rsna.org/dataset/5](https://mira.rsna.org/dataset/5)), locate the data request link, and fill in their email address. Upon approval, the data provider will supply specific download instructions.

The clinical decision-making task data from the Beth Israel Deaconess Medical Center used for downstream external validation are available on PhysioNet as the MIMIC-IV-EXT-MDS-ED dataset (access requires certification for researchers, [https://physionet.org/content/multimodal-emergency-benchmark/1.0.0/](https://physionet.org/content/multimodal-emergency-benchmark/1.0.0/)). The remaining MIMIC-related datasets can be obtained from the original sources according to their respective access rules. Data from other participating sites (SYSMH, GTCMH) are not publicly available and require access through data use agreements with each respective institution.

## File Structure

```
huggingface_model/
|-- pytorch_model.bin           # Pretrained weights (9.8GB)
|-- config.json                 # Model config
|-- modeling_beit3_ed.py        # Model definition
|-- tokenizer_config.json       # Tokenizer config
|-- README.md                   # This file
|-- requirements.txt            # Dependencies
|-- LICENSE                     # Apache 2.0 License
|-- example_usage.py            # Example usage
`-- datasets_scripts/           # Dataset scripts
    |-- DATASETS_GUIDE.md       # Dataset guide
    |-- train_template.py       # Training template
    |-- early_triage/           # Early triage scripts
    |-- prognosis_prediction/   # Prognosis prediction scripts
    `-- MDs/                    # Full-process decision scripts
```

## Important Notes

### Scope

- Research and development
- Model evaluation and benchmarking
- Education and training
- **Not for direct clinical decision-making**

### Limitations

1. **Clinical validation required**: predictions must be interpreted with clinical judgment
2. **Data shift**: trained on specific datasets; distribution bias may exist
3. **Zero-shot variability**: performance may vary across hospitals
4. **Ethics**: protect patient privacy and follow medical ethics

## License

This model is released under the [Apache 2.0 License](LICENSE).

## Contributing

Contributions are welcome. Please open an Issue or Pull Request.

## Acknowledgements

- **Datasets**: MIMIC-IV-ED, SYSMH, GTCMH
- **Base Model**: [BEiT](https://github.com/microsoft/unilm/tree/master/beit)
- **Libraries**: [timm](https://github.com/huggingface/pytorch-image-models), [transformers](https://github.com/huggingface/transformers)
