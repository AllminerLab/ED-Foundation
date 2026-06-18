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
