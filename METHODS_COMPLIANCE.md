# Methods Compliance Notes

This repository is organized so users can map the released code to the Methods
section of the manuscript without requiring access to private clinical data.

## What

- `methods_protocol.json`: machine-readable protocol covering study design,
  pretraining settings, downstream dataset schemas, split rules, task labels,
  missingness handling, evaluation metrics, and 7-run reporting rules.
- `methods_reproducibility.py`: one entry-point script for generating all
  Methods-related intermediate artifacts under a single artifact directory
  (`artifacts/methods/` by default).
- `modeling_beit3_ed.py`: model definition and linear-probing task heads.
- `datasets_scripts/`: task-specific training and inference scripts.

## Mapping to Manuscript Methods

| Methods item | Repository support |
| --- | --- |
| Multi-center retrospective design, ethics, and no prospective intervention | Recorded in `methods_protocol.json` |
| Two-stage pretraining: masked modeling and contrastive learning | Pretraining hyperparameters and losses recorded in `methods_protocol.json` |
| BEIT-3 architectural specification | `config.json`, `methods_protocol.json`, `modeling_beit3_ed.py` |
| Dataset-specific inputs, labels, split ratios, zero-shot design | `methods_protocol.json` |
| Template-based input text construction | `methods_reproducibility.py prepare-dataset` |
| Missing text fields as empty strings and missingness indicators | `methods_reproducibility.py prepare-dataset` |
| Patient-level or visit-level split generation | `methods_reproducibility.py prepare-dataset` |
| Linear probing with frozen backbone | `modeling_beit3_ed.py` and `datasets_scripts/train_template.py` |
| Class-weight balanced cross-entropy | `modeling_beit3_ed.py` optional `class_weights` argument |
| Seven-run metric aggregation | `methods_reproducibility.py evaluate-runs` |
| Binary threshold 0.5 and multiclass argmax | `methods_reproducibility.py evaluate-runs` |
| Sensitivity, precision, F1, macro-F1, high-risk recall | `methods_reproducibility.py evaluate-runs` |
| Calibration curves and Expected Calibration Error | `methods_reproducibility.py evaluate-runs` |
| Confusion matrix run closest to the 7-run mean | `methods_reproducibility.py evaluate-runs` |
| Exploratory non-urgent triage resource aggregation | `methods_reproducibility.py simulate-triage-cost` |

## Artifact Convention

All generated intermediate files should be written under one directory:

```bash
python methods_reproducibility.py \
  --artifact-dir artifacts/methods \
  prepare-dataset \
  --dataset-name SYSMH-S-Triage \
  --input-csv /path/to/sysmh_s_triage.csv
```

This produces:

```text
artifacts/methods/SYSMH-S-Triage/
├── manifest.json
├── processed.csv
├── train.csv
├── val.csv
└── test.csv
```

For 7-run evaluation, place one prediction CSV per seed in a directory. Each file
must contain:

- Binary task: `y_true,y_prob`
- Multiclass task: `y_true,y_pred`

Then run:

```bash
python methods_reproducibility.py \
  --artifact-dir artifacts/methods \
  evaluate-runs \
  --dataset-name SYSMH-ED-Outcome \
  --predictions-dir /path/to/prediction_csvs
```

For Chinese triage resource aggregation:

```bash
python methods_reproducibility.py \
  --artifact-dir artifacts/methods \
  simulate-triage-cost \
  --dataset-name SYSMH-S-Triage \
  --predictions-dir /path/to/prediction_csvs
```

The cost simulation is exploratory and should not be interpreted as actual cost
savings, operational improvement, or validated health-economic benefit.

## Data Availability Boundary

The repository does not include restricted clinical data, pretrained weights, or
downstream fine-tuned weights. Private SYSMH and GTCMH data require institutional
data-use agreements. MIMIC data require PhysioNet credentialed access. The
pretrained weights and downstream fine-tuned weights will be released after
publication of the manuscript.
