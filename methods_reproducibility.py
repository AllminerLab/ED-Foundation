#!/usr/bin/env python3
"""
Utilities that keep Methods-related preprocessing and evaluation artifacts in one
place for repository review.

The script does not download restricted clinical data. It expects CSV files that
already follow the schemas in methods_protocol.json and writes every generated
intermediate file under one artifact directory.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


DEFAULT_SEEDS = [2026, 2027, 2028, 2029, 2030, 2031, 2032]
EXCLUSION_LABEL = -999


def load_protocol(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_artifact_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def split_spec_to_ratios(spec: str) -> Optional[Tuple[float, float, float]]:
    if spec == "external_test_only":
        return None
    parts = [float(x) for x in spec.split(":")]
    if len(parts) != 3:
        raise ValueError(f"Unsupported split specification: {spec}")
    total = sum(parts)
    return parts[0] / total, parts[1] / total, parts[2] / total


def stable_group_split(
    df: pd.DataFrame,
    group_col: str,
    split_spec: str,
    seed: int,
) -> pd.Series:
    ratios = split_spec_to_ratios(split_spec)
    if ratios is None:
        return pd.Series(["test"] * len(df), index=df.index)

    rng = random.Random(seed)
    groups = list(pd.Series(df[group_col].astype(str).unique()).dropna())
    rng.shuffle(groups)
    n = len(groups)
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    train_groups = set(groups[:n_train])
    val_groups = set(groups[n_train : n_train + n_val])

    def assign(group: object) -> str:
        value = str(group)
        if value in train_groups:
            return "train"
        if value in val_groups:
            return "val"
        return "test"

    return df[group_col].map(assign)


def missing_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def value_or_empty(row: pd.Series, column: str) -> str:
    if column not in row or missing_value(row[column]):
        return ""
    return str(row[column])


def format_vitals(row: pd.Series) -> str:
    fields = [
        ("SBP", "systolic_bp"),
        ("DBP", "diastolic_bp"),
        ("temperature", "temperature"),
        ("heart rate", "heart_rate"),
        ("respiratory rate", "respiratory_rate"),
        ("oxygen saturation", "oxygen_saturation"),
        ("blood glucose", "blood_glucose"),
        ("weight", "weight"),
        ("height", "height"),
        ("consciousness", "consciousness"),
        ("pain score", "pain_score"),
    ]
    chunks = []
    for label, column in fields:
        if column in row and not missing_value(row[column]):
            chunks.append(f"{label}: {row[column]}")
    return "; ".join(chunks)


def construct_text(row: pd.Series, template: str) -> str:
    if "text" in row and not missing_value(row["text"]):
        return str(row["text"])

    if template == "triage_cn":
        return (
            f"Chief complaint: {value_or_empty(row, 'chief_complaint')}. "
            f"Age: {value_or_empty(row, 'age')}. Sex: {value_or_empty(row, 'sex')}. "
            f"Arrival mode: {value_or_empty(row, 'arrival_mode')}. "
            f"Early vital signs: {format_vitals(row)}."
        )
    if template == "triage_en":
        return (
            f"Chief complaint: {value_or_empty(row, 'chief_complaint')}. "
            f"Age: {value_or_empty(row, 'age')}. Sex: {value_or_empty(row, 'sex')}. "
            f"Arrival mode: {value_or_empty(row, 'arrival_mode')}. "
            f"Early vital signs: {format_vitals(row)}."
        )
    if template == "clinical_text_cn":
        return (
            f"Chief complaint: {value_or_empty(row, 'chief_complaint')}. "
            f"History of present illness: {value_or_empty(row, 'history_present_illness')}. "
            f"Past medical history: {value_or_empty(row, 'past_medical_history')}. "
            f"Primary diagnosis: {value_or_empty(row, 'primary_diagnosis')}. "
            f"Age: {value_or_empty(row, 'age')}. Sex: {value_or_empty(row, 'sex')}. "
            f"Early vital signs: {format_vitals(row)}."
        )
    if template == "clinical_text_en":
        return (
            f"Chief complaint: {value_or_empty(row, 'chief_complaint')}. "
            f"History of present illness: {value_or_empty(row, 'history_present_illness')}. "
            f"Past medical history: {value_or_empty(row, 'past_medical_history')}. "
            f"Primary diagnosis: {value_or_empty(row, 'primary_diagnosis')}. "
            f"Age: {value_or_empty(row, 'age')}. Sex: {value_or_empty(row, 'sex')}. "
            f"Early vital signs: {format_vitals(row)}."
        )
    raise ValueError(f"Unknown input template: {template}")


def add_missing_indicators(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[f"{column}_missing"] = out[column].isna() | (out[column].astype(str).str.strip() == "")
            out[column] = out[column].fillna("")
    return out


def prepare_dataset(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.protocol)
    dataset_cfg = protocol["datasets"][args.dataset_name]
    artifact_dir = ensure_artifact_dir(args.artifact_dir / args.dataset_name)

    df = pd.read_csv(args.input_csv)
    required = dataset_cfg.get("required_columns", [])
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{args.dataset_name} is missing required columns: {missing}")

    if dataset_cfg.get("exclusion_label") is not None:
        label_columns = list(dataset_cfg.get("tasks", {}).keys())
        for label_col in label_columns:
            if label_col in df.columns:
                df = df[df[label_col] != dataset_cfg["exclusion_label"]].copy()

    df = add_missing_indicators(df, required)
    df["constructed_text"] = df.apply(
        lambda row: construct_text(row, dataset_cfg["input_template"]), axis=1
    )

    group_col = "patient_id" if dataset_cfg.get("split_level") == "patient" else "visit_id"
    if group_col not in df.columns:
        raise ValueError(f"Split level requires column '{group_col}'")
    df["split"] = stable_group_split(df, group_col, dataset_cfg["split"], args.seed)

    processed_path = artifact_dir / "processed.csv"
    df.to_csv(processed_path, index=False)
    for split_name, split_df in df.groupby("split"):
        split_df.to_csv(artifact_dir / f"{split_name}.csv", index=False)

    completeness = {}
    for column in required:
        if column in df.columns:
            completeness[column] = float(1.0 - df[f"{column}_missing"].mean())

    manifest = {
        "dataset_name": args.dataset_name,
        "source_csv": str(args.input_csv),
        "artifact_dir": str(artifact_dir),
        "processed_csv": str(processed_path),
        "split_counts": df["split"].value_counts().to_dict(),
        "required_columns": required,
        "completeness": completeness,
        "template": dataset_cfg["input_template"],
        "split_seed": args.seed,
    }
    write_json(artifact_dir / "manifest.json", manifest)


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * sensitivity * precision / (sensitivity + precision) if (sensitivity + precision) else 0.0
    return {
        "f1": float(f1),
        "sensitivity": float(sensitivity),
        "precision": float(precision),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_sensitivity": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def high_risk_recall(y_true: np.ndarray, y_pred: np.ndarray, high_risk_labels: List[int]) -> float:
    high = set(high_risk_labels)
    true_binary = np.array([1 if int(x) in high else 0 for x in y_true])
    pred_binary = np.array([1 if int(x) in high else 0 for x in y_pred])
    return float(recall_score(true_binary, pred_binary, zero_division=0))


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    default_bins: int = 10,
    small_bins: int = 8,
    small_threshold: int = 500,
) -> Dict:
    bins = small_bins if len(y_true) < small_threshold else default_bins
    order = np.argsort(y_prob)
    y_true_sorted = y_true[order]
    y_prob_sorted = y_prob[order]
    chunks = np.array_split(np.arange(len(y_true_sorted)), bins)
    ece = 0.0
    points = []
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        observed = float(np.mean(y_true_sorted[chunk]))
        confidence = float(np.mean(y_prob_sorted[chunk]))
        weight = len(chunk) / len(y_true_sorted)
        ece += weight * abs(observed - confidence)
        points.append({"n": int(len(chunk)), "observed": observed, "confidence": confidence})
    return {"ece": float(ece), "bins": bins, "points": points}


def read_run_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "y_true" not in df.columns:
        raise ValueError(f"{path} must include a y_true column")
    if "y_pred" not in df.columns and "y_prob" not in df.columns:
        raise ValueError(f"{path} must include y_pred for multiclass or y_prob for binary")
    return df


def evaluate_runs(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.protocol)
    dataset_cfg = protocol["datasets"][args.dataset_name]
    artifact_dir = ensure_artifact_dir(args.artifact_dir / args.dataset_name / "evaluation")

    run_files = sorted(args.predictions_dir.glob("*.csv"))
    if not run_files:
        raise ValueError(f"No CSV prediction files found in {args.predictions_dir}")

    summaries = []
    calibration = {}
    for run_file in run_files:
        df = read_run_predictions(run_file)
        y_true = df["y_true"].to_numpy()
        if args.task_name:
            task_cfg = dataset_cfg.get("tasks", {}).get(args.task_name, {})
        else:
            task_cfg = dataset_cfg

        is_binary = len(task_cfg.get("classes", [])) == 2 or "y_prob" in df.columns
        if is_binary:
            y_prob = df["y_prob"].to_numpy(dtype=float)
            metrics = binary_metrics(
                y_true.astype(int), y_prob, protocol["downstream_protocol"]["binary_threshold"]
            )
            calibration[run_file.stem] = expected_calibration_error(
                y_true.astype(int),
                y_prob,
                protocol["downstream_protocol"]["calibration_bins_default"],
                protocol["downstream_protocol"]["calibration_bins_small_test"],
                protocol["downstream_protocol"]["small_test_threshold"],
            )
        else:
            y_pred = df["y_pred"].to_numpy()
            metrics = multiclass_metrics(y_true, y_pred)
            if dataset_cfg.get("high_risk_labels"):
                metrics["high_risk_recall"] = high_risk_recall(
                    y_true, y_pred, dataset_cfg["high_risk_labels"]
                )

        metrics["run"] = run_file.stem
        summaries.append(metrics)

    summary_df = pd.DataFrame(summaries)
    metric_cols = [c for c in summary_df.columns if c != "run"]
    mean_metrics = {c: float(summary_df[c].mean()) for c in metric_cols if pd.api.types.is_numeric_dtype(summary_df[c])}
    if "sensitivity" in mean_metrics and "precision" in mean_metrics:
        sensitivity = mean_metrics["sensitivity"]
        precision = mean_metrics["precision"]
        mean_metrics["f1"] = (
            2 * sensitivity * precision / (sensitivity + precision)
            if (sensitivity + precision)
            else 0.0
        )

    primary = args.primary_metric or dataset_cfg.get("primary_metric") or "f1"
    if primary in summary_df:
        closest_idx = (summary_df[primary] - mean_metrics[primary]).abs().idxmin()
        closest_run = summary_df.loc[closest_idx, "run"]
    else:
        closest_run = summary_df.loc[0, "run"]

    summary_df.to_csv(artifact_dir / "run_metrics.csv", index=False)
    write_json(
        artifact_dir / "summary.json",
        {
            "dataset_name": args.dataset_name,
            "task_name": args.task_name,
            "n_runs": len(run_files),
            "mean_metrics": mean_metrics,
            "closest_run_for_confusion_matrix": closest_run,
            "calibration": calibration,
            "protocol": {
                "binary_threshold": protocol["downstream_protocol"]["binary_threshold"],
                "multiclass_rule": protocol["downstream_protocol"]["multiclass_rule"],
            },
        },
    )


def simulate_triage_cost(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.protocol)
    dataset_cfg = protocol["datasets"][args.dataset_name]
    if dataset_cfg["task_family"] != "early_triage":
        raise ValueError("Cost simulation is only defined for early triage datasets")
    if args.dataset_name == "MIMIC-IV-ED-Triage":
        raise ValueError("Cost simulation is not applied to MIMIC-IV-ED-Triage")

    artifact_dir = ensure_artifact_dir(args.artifact_dir / args.dataset_name / "cost_simulation")
    run_files = sorted(args.predictions_dir.glob("*.csv"))
    non_urgent = set(args.non_urgent_labels or [3, 4])
    per_visit_cost = args.per_visit_cost_cny or protocol["downstream_protocol"]["non_urgent_cost_cny"]

    rows = []
    for run_file in run_files:
        df = read_run_predictions(run_file)
        if "y_pred" not in df.columns:
            raise ValueError(f"{run_file} must include y_pred for triage cost simulation")
        correct_non_urgent = df[
            df["y_true"].astype(int).isin(non_urgent) & df["y_pred"].astype(int).isin(non_urgent)
        ]
        rows.append(
            {
                "run": run_file.stem,
                "correct_non_urgent_count": int(len(correct_non_urgent)),
                "simulated_resource_cny": float(len(correct_non_urgent) * per_visit_cost),
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(artifact_dir / "cost_simulation_runs.csv", index=False)
    write_json(
        artifact_dir / "summary.json",
        {
            "dataset_name": args.dataset_name,
            "per_visit_cost_cny": per_visit_cost,
            "non_urgent_labels": sorted(non_urgent),
            "mean_correct_non_urgent_count": float(result["correct_non_urgent_count"].mean()),
            "mean_simulated_resource_cny": float(result["simulated_resource_cny"].mean()),
            "interpretation": "Exploratory retrospective aggregation only; not evidence of actual cost savings.",
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ED-Foundation Methods reproducibility utilities")
    parser.add_argument("--protocol", type=Path, default=Path("methods_protocol.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts") / "methods")

    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prepare-dataset", help="Construct text, missing indicators, and splits")
    prep.add_argument("--dataset-name", required=True)
    prep.add_argument("--input-csv", type=Path, required=True)
    prep.add_argument("--seed", type=int, default=DEFAULT_SEEDS[0])
    prep.set_defaults(func=prepare_dataset)

    eval_parser = subparsers.add_parser("evaluate-runs", help="Aggregate 7-run metrics and calibration")
    eval_parser.add_argument("--dataset-name", required=True)
    eval_parser.add_argument("--task-name")
    eval_parser.add_argument("--predictions-dir", type=Path, required=True)
    eval_parser.add_argument("--primary-metric")
    eval_parser.set_defaults(func=evaluate_runs)

    cost = subparsers.add_parser("simulate-triage-cost", help="Run exploratory non-urgent triage cost aggregation")
    cost.add_argument("--dataset-name", required=True)
    cost.add_argument("--predictions-dir", type=Path, required=True)
    cost.add_argument("--per-visit-cost-cny", type=float)
    cost.add_argument("--non-urgent-labels", type=int, nargs="*")
    cost.set_defaults(func=simulate_triage_cost)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
