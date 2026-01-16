import argparse
import json
import re
from pathlib import Path

import pandas as pd


DEFAULT_RESULTS_FILENAME = "final_summary_results.csv"
DEFAULT_STATS_FILENAME = "final_summary_results_stats.csv"
CONFIG_COLS = ["Rank", "Alpha", "LR", "Target_Scale"]
KNOWN_METRICS = [
    "RMSE",
    "Invalid_Ratio",
    "Final_Invalid_Ratio",
    "Best_Train_Loss",
    "Best_Val_Loss",
    "Best_Val_RMSE",
    "Best_Invalid_Ratio",
    "Best_Final_Invalid_Ratio",
    "Best_Greedy_Success_Rate",
    "Best_Retry_Success_Rate",
    "Best_Fallback_Rate",
    "Test_Greedy_Success_Rate",
    "Test_Retry_Success_Rate",
    "Test_Fallback_Rate",
    "Test_Loss",
]
ALPHA_CANDIDATES = ["Alpha", "Lora_Alpha", "LoRA_Alpha", "lora_alpha"]
CONFIG_DIR_RE = re.compile(
    r"^epochs_(?P<epochs>\d+)_rank(?P<rank>\d+)_alpha(?P<alpha>\d+)_lr(?P<lr>[^_]+)_drop(?P<drop>[^_]+)$"
)
SCALE_DIR_RE = re.compile(r"^scale_(?P<scale>log|linear)$")
DATA_SEED_RE = re.compile(r"^data(?P<data_seed>\d+)$")
MODEL_SEED_RE = re.compile(r"^seed_(?P<model_seed>\d+)$")


def _get_metric_cols(df: pd.DataFrame) -> list[str]:
    return [col for col in KNOWN_METRICS if col in df.columns]


def _make_key(row: pd.Series) -> tuple:
    return tuple("<NA>" if pd.isna(value) else value for value in row)


def _order_map(df: pd.DataFrame, cols: list[str]) -> dict[tuple, int]:
    order = {}
    for _, row in df[cols].iterrows():
        key = _make_key(row)
        if key not in order:
            order[key] = len(order)
    return order


def _apply_config_order(
    df: pd.DataFrame, config_order: dict[tuple, int], cols: list[str]
) -> pd.DataFrame:
    ordered = df.copy()
    ordered["_config_order"] = [
        config_order[_make_key(row)] for _, row in ordered[cols].iterrows()
    ]
    ordered = ordered.sort_values(["_config_order"]).drop(columns=["_config_order"])
    return ordered


def summarize_results(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    if "Alpha" not in df.columns:
        for candidate in ALPHA_CANDIDATES:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "Alpha"})
                break
    required_cols = {"Data_Seed", *CONFIG_COLS}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    metric_cols = _get_metric_cols(df)
    if not metric_cols:
        raise ValueError("No metric columns found to summarize.")

    config_order = _order_map(df, CONFIG_COLS)
    data_seed_order = _order_map(df, ["Data_Seed"])

    per_seed_grouped = (
        df.groupby(["Data_Seed", *CONFIG_COLS], dropna=False, sort=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    per_seed_grouped.columns = [
        "_".join(col).strip("_") for col in per_seed_grouped.columns.to_flat_index()
    ]
    per_seed_grouped["n_runs"] = (
        df.groupby(["Data_Seed", *CONFIG_COLS], dropna=False, sort=False)
        .size()
        .values
    )

    per_seed_grouped["_seed_order"] = [
        data_seed_order[_make_key(pd.Series([seed]))]
        for seed in per_seed_grouped["Data_Seed"]
    ]
    per_seed_grouped["_config_order"] = [
        config_order[_make_key(row)] for _, row in per_seed_grouped[CONFIG_COLS].iterrows()
    ]
    per_seed_grouped = per_seed_grouped.sort_values(
        ["_seed_order", "_config_order"]
    ).drop(columns=["_seed_order", "_config_order"])

    overall_grouped = (
        df.groupby(CONFIG_COLS, dropna=False, sort=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    overall_grouped.columns = [
        "_".join(col).strip("_") for col in overall_grouped.columns.to_flat_index()
    ]
    overall_grouped["n_seeds"] = (
        df.groupby(CONFIG_COLS, dropna=False, sort=False)["Data_Seed"]
        .nunique()
        .values
    )
    overall_grouped = _apply_config_order(overall_grouped, config_order, CONFIG_COLS)

    return per_seed_grouped, overall_grouped


def _parse_config_from_dir(dir_name: str) -> dict:
    match = CONFIG_DIR_RE.match(dir_name)
    if not match:
        return {}
    return {
        "Epochs": int(match.group("epochs")),
        "Rank": int(match.group("rank")),
        "Alpha": int(match.group("alpha")),
        "LR": match.group("lr"),
        "Dropout": float(match.group("drop")),
    }


def _find_target_scale(parts: tuple[str, ...]) -> str | None:
    for part in parts:
        match = SCALE_DIR_RE.match(part)
        if match:
            return match.group("scale")
    return None


def _find_seed_from_parts(parts: tuple[str, ...], regex: re.Pattern) -> int | None:
    for part in parts:
        match = regex.match(part)
        if match:
            return int(next(iter(match.groupdict().values())))
    return None


def _metric_at_epoch(metrics: dict, key: str, epoch: int | None) -> float | None:
    if not epoch:
        return None
    values = metrics.get(key)
    if not values:
        return None
    if 1 <= epoch <= len(values):
        return values[epoch - 1]
    return None


def collect_final_summary(results_root: Path) -> pd.DataFrame:
    rows = []
    for test_metrics_path in results_root.rglob("test_metrics.json"):
        config_dir = test_metrics_path.parent
        config = _parse_config_from_dir(config_dir.name)
        if not config:
            continue

        parts = tuple(test_metrics_path.parts)
        data_seed = _find_seed_from_parts(parts, DATA_SEED_RE)
        model_seed = _find_seed_from_parts(parts, MODEL_SEED_RE)
        target_scale = _find_target_scale(parts)
        experiment = results_root.name
        if results_root in test_metrics_path.parents:
            relative = test_metrics_path.relative_to(results_root)
            if relative.parts and relative.parts[0].startswith("data"):
                experiment = f"{results_root.name}_{relative.parts[0]}"
            elif len(relative.parts) > 1 and relative.parts[1].startswith("data"):
                experiment = f"{relative.parts[0]}_{relative.parts[1]}"
            else:
                experiment = relative.parts[0] if relative.parts else results_root.name

        with open(test_metrics_path, "r") as fp:
            test_metrics = json.load(fp)

        best_model_info = {}
        best_model_path = config_dir / "best_model_info.json"
        if best_model_path.exists():
            with open(best_model_path, "r") as fp:
                best_model_info = json.load(fp)

        validation_metrics = {}
        validation_metrics_path = config_dir / "validation_metrics.json"
        if validation_metrics_path.exists():
            with open(validation_metrics_path, "r") as fp:
                validation_metrics = json.load(fp)

        best_epoch = best_model_info.get("epoch")
        best_train_loss = _metric_at_epoch(validation_metrics, "train_loss", best_epoch)
        best_val_loss = best_model_info.get("val_loss") or _metric_at_epoch(
            validation_metrics, "val_loss", best_epoch
        )
        best_val_rmse = best_model_info.get("val_rmse") or _metric_at_epoch(
            validation_metrics, "val_rmse", best_epoch
        )
        best_invalid_ratio = best_model_info.get("invalid_ratio") or _metric_at_epoch(
            validation_metrics, "invalid_answer_ratio", best_epoch
        )
        best_final_invalid_ratio = best_model_info.get(
            "final_invalid_ratio"
        ) or _metric_at_epoch(validation_metrics, "final_invalid_answer_ratio", best_epoch)
        best_greedy_success_rate = _metric_at_epoch(
            validation_metrics, "greedy_success_rate", best_epoch
        )
        best_retry_success_rate = _metric_at_epoch(
            validation_metrics, "retry_success_rate", best_epoch
        )
        best_fallback_rate = _metric_at_epoch(
            validation_metrics, "fallback_rate", best_epoch
        )

        rows.append(
            {
                "Experiment": experiment,
                "Data_Seed": data_seed,
                "Model_Seed": model_seed,
                "Target_Scale": target_scale,
                **config,
                "RMSE": test_metrics.get("test_rmse"),
                "Invalid_Ratio": test_metrics.get("invalid_answer_ratio"),
                "Final_Invalid_Ratio": test_metrics.get("final_invalid_answer_ratio"),
                "Test_Greedy_Success_Rate": test_metrics.get("greedy_success_rate"),
                "Test_Retry_Success_Rate": test_metrics.get("retry_success_rate"),
                "Test_Fallback_Rate": test_metrics.get("fallback_rate"),
                "Best_Train_Loss": best_train_loss,
                "Best_Val_Loss": best_val_loss,
                "Best_Val_RMSE": best_val_rmse,
                "Best_Invalid_Ratio": best_invalid_ratio,
                "Best_Final_Invalid_Ratio": best_final_invalid_ratio,
                "Best_Greedy_Success_Rate": best_greedy_success_rate,
                "Best_Retry_Success_Rate": best_retry_success_rate,
                "Best_Fallback_Rate": best_fallback_rate,
                "Best_Epoch": best_epoch,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No test_metrics.json files found under {results_root}")

    # Order: config group first, then data seed within each config.
    config_order = _order_map(df, CONFIG_COLS)
    df["_config_order"] = [
        config_order[_make_key(row)] for _, row in df[CONFIG_COLS].iterrows()
    ]
    df = df.sort_values(["_config_order", "Data_Seed", "Model_Seed"]).drop(
        columns=["_config_order"]
    )

    # Format non-hyperparam numeric metrics to 4 decimals.
    metric_cols = [col for col in KNOWN_METRICS if col in df.columns]
    for col in metric_cols:
        df[col] = df[col].apply(
            lambda val: round(val, 4) if pd.notna(val) else val
        )

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize grid search results by seed and overall config stats."
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Collect per-run metrics into final_summary_results.csv",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to final_summary_results.csv",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).with_name("results"),
        help="Root results folder containing experiment folders (dates).",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment folder name under results root (e.g., 240101).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path for aggregated stats",
    )
    parser.add_argument(
        "--out-by-seed",
        type=Path,
        default=None,
        help="Optional output CSV path for per-seed aggregated stats",
    )
    args = parser.parse_args()

    if args.collect:
        collect_root = args.results_root
        if args.experiment:
            collect_root = collect_root / args.experiment
        out_csv = args.out or (collect_root / DEFAULT_RESULTS_FILENAME)
        collected = collect_final_summary(collect_root)
        collected.to_csv(out_csv, index=False)
        print("Saved collected results to:", out_csv)
        return

    if args.csv is not None:
        csv_path = args.csv
    elif args.experiment is not None:
        csv_path = (
            args.results_root / args.experiment / DEFAULT_RESULTS_FILENAME
        )
    else:
        csv_path = Path(__file__).with_name(DEFAULT_RESULTS_FILENAME)

    if not csv_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")

    out_dir = csv_path.parent
    out_stats = args.out or (out_dir / DEFAULT_STATS_FILENAME)
    out_by_seed = args.out_by_seed

    per_seed_summary, overall_summary = summarize_results(csv_path)
    if out_by_seed is not None:
        per_seed_summary.to_csv(out_by_seed, index=False)
        print("Saved per-seed summary to:", out_by_seed)
    overall_summary.to_csv(out_stats, index=False)
    print("Saved overall summary to:", out_stats)


if __name__ == "__main__":
    main()
