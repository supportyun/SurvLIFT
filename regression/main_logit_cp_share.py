import argparse
import json
import os
from datetime import datetime

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import numpy as np
import pandas as pd
import torch
from huggingface_hub import login
from sklearn.model_selection import train_test_split
from transformers import set_seed

from data_utils_share import *  # noqa: F403  # NOTE: 공부용 메모 - 전부 import.
from llama_finetuner_logit_cp_share import LlamaFinetuner

login(token=os.getenv("HF_TOKEN"))


def prepare_data(data_seed, target_scale):
    # NOTE: 공부용 메모 - 스크립트 위치 기준 절대경로 사용.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    file_path = os.path.join(
        project_root, "data", f"target_data_for_aft_seed{data_seed}_{target_scale}.csv"
    )

    print(f"Loading data from: {file_path}")
    if not os.path.exists(file_path):
        data_dir = os.path.dirname(file_path)
        if os.path.exists(data_dir):
            print(f"Files in data dir: {os.listdir(data_dir)}")
        else:
            print(f"Data directory does not exist: {data_dir}")
        raise FileNotFoundError(f"Data file not found: {file_path}")

    df = pd.read_csv(file_path)

    # NOTE: 공부용 메모 - event/censor 분리 후 8:1:1로 split.
    event_data = df[df["status"] == 1]
    censor_data = df[df["status"] == 0]
    event_train, event_temp = train_test_split(event_data, test_size=0.2, random_state=42)
    event_validate, event_test = train_test_split(
        event_temp, test_size=0.5, random_state=42
    )
    censor_train, censor_temp = train_test_split(
        censor_data, test_size=0.2, random_state=42
    )
    censor_validate, censor_test = train_test_split(
        censor_temp, test_size=0.5, random_state=42
    )

    # NOTE: 공부용 메모 - frac=1.0 => 전체 shuffle.
    train = pd.concat([event_train, censor_train]).sample(frac=1, random_state=42)
    validate = pd.concat([event_validate, censor_validate]).sample(frac=1, random_state=42)
    test = pd.concat([event_test, censor_test]).sample(frac=1, random_state=42)

    return train, validate, test


def build_output_dir(args):
    experiment_date = args.experiment_date or datetime.now().strftime("%y%m%d")
    base_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"results/{experiment_date}/data{args.data_seed}/seed_{args.seed}/scale_{args.target_scale}",
    )
    folder_name = (
        f"epochs_{args.epochs}_"
        f"rank{args.r}_"
        f"alpha{args.lora_alpha}_"
        f"lr{args.lr}_"
        f"drop{args.lora_dropout}"
    )
    output_dir = os.path.join(base_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def compute_success_rates(logs):
    # NOTE: 공부용 메모 - generate 로그에서 greedy/retry/fallback 비율 계산.
    if not logs:
        return 0.0, 0.0, 0.0
    total = len(logs)
    n_greedy = sum(1 for log in logs if log.get("attempt") == "greedy" and log.get("is_valid"))
    n_retry = sum(
        1
        for log in logs
        if str(log.get("attempt", "")).startswith("retry") and log.get("is_valid")
    )
    n_fallback = sum(1 for log in logs if log.get("attempt") == "fallback_stat")
    return n_greedy / total, n_retry / total, n_fallback / total


def main(args):
    print(f"Prepare data (Seed: {args.data_seed})...")
    set_seed(args.seed)

    # 1) Load and split data.
    train, validate, test = prepare_data(args.data_seed, args.target_scale)

    # 2) Convert to prompts.
    # NOTE: 공부용 메모 - data2text_cp2 수정된 프롬프트 사용.
    train_prompts = df2prompts(
        train, data2text_cp2, "", "", target_scale=args.target_scale
    )  # noqa: F405
    val_prompts = df2prompts(
        validate, data2text_cp2, "", "", target_scale=args.target_scale
    )  # noqa: F405
    test_prompts = df2prompts(
        test, data2text_cp2, "", "", target_scale=args.target_scale
    )  # noqa: F405
    print("Save prompts...")

    output_dir = build_output_dir(args)
    train_js_path = os.path.join(output_dir, "synthetic_prompts_train.jsonl")
    val_js_path = os.path.join(output_dir, "synthetic_prompts_val.jsonl")
    test_js_path = os.path.join(output_dir, "synthetic_prompts_test.jsonl")

    # NOTE: 공부용 메모 - 리스트를 \n 기준으로 jsonl 저장.
    write_jsonl("\n".join(train_prompts), train_js_path)  # noqa: F405
    write_jsonl("\n".join(val_prompts), val_js_path)  # noqa: F405
    write_jsonl("\n".join(test_prompts), test_js_path)  # noqa: F405

    # 3) Build fallback means for invalid predictions.
    # NOTE: 공부용 메모 - fallback은 train 기준 그룹 평균 + global mean.
    print("Building group means for fallback...")
    group_means_dict = build_time_group_means(train, target_col="target_time")  # noqa: F405
    train_mean = round(train["target_time"].mean(), 2)
    print(f"Train Global Mean: {train_mean:.2f}")

    # 4) Train.
    # NOTE: 공부용 메모 - LoRA 설정 및 4bit 양자화 관련 인자들.
    print("Start train...")
    finetuner = LlamaFinetuner(
        model_name=args.model_name,
        device=args.device,
        output_dir=output_dir,
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        seed=args.seed,
    )

    finetuner.train(
        train_js_path,
        val_js_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        saving_checkpoint=args.saving_checkpoint,
        unique_validate_df=None,
        interp_method=args.interp_method,
        min_g=args.min_g,
        fallback_means=group_means_dict,
    )

    # 5) Evaluate.
    # NOTE: 공부용 메모 - load_model2로 안전 로딩.
    print("Start evaluate...")
    finetuner.load_model2()

    test_prompts_loaded = extract_prompts(test_js_path, "")  # noqa: F405
    test_completions = extract_completion(test_js_path)  # noqa: F405
    test_truth = [float(s.split("@@@", 1)[0].strip()) for s in test_completions]

    # NOTE: 공부용 메모 - valid_mean은 global mean, fallback은 group mean.
    if args.target_scale == "log":
        clip_min, clip_max = -5.0, 10.0
    else:
        clip_min, clip_max = 0.0, 120.0

    ans, invalid_ratio, final_invalid_ratio, test_logs = finetuner.generate(
        text_lst=test_prompts_loaded,
        max_token=10,
        batch_size=args.batch_size,
        valid_mean=train_mean,
        fallback_means=group_means_dict,
        clip_min=clip_min,
        clip_max=clip_max,
        return_logs=True,
    )

    test_greedy_rate, test_retry_rate, test_fallback_rate = compute_success_rates(
        test_logs
    )
    log_df = pd.DataFrame(test_logs)
    log_path = os.path.join(output_dir, "test_predictions_detailed_logs.csv")
    log_df.to_csv(log_path, index=False, encoding="utf-8-sig")
    print(f"Saved test logs to {log_path}")
    print("Predictions sample:", ans[:5])

    test_dataset = finetuner.prepare_data(test_js_path)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)
    test_loss = finetuner.validate(test_loader)

    mse = np.mean((np.array(test_truth) - np.array(ans)) ** 2)
    rmse = np.sqrt(mse)
    print(f"Test RMSE: {rmse:.4f}")

    result_df = pd.DataFrame(
        {
            "id": test["id"].values,
            "true_target_time": test_truth,
            "pred_time": ans,
            "observed_time": test["time"].values,
            "status": test["status"].values,
        }
    )
    result_df.to_csv(os.path.join(output_dir, "prediction_results.csv"), index=False)

    # 6) Save test metrics for aggregation.
    # NOTE: 공부용 메모 - summary CSV는 summarize_grid_results.py에서 재구성.
    metrics = {
        "test_rmse": rmse,
        "invalid_answer_ratio": invalid_ratio,
        "final_invalid_answer_ratio": final_invalid_ratio,
        "greedy_success_rate": test_greedy_rate,
        "retry_success_rate": test_retry_rate,
        "fallback_rate": test_fallback_rate,
    }
    json_path = os.path.join(output_dir, "test_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("All evaluation finished!")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_seed", type=int, default=1, help="Data file seed (1-10)")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--load_in_4bit", type=bool, default=True)
    parser.add_argument("--bnb_4bit_use_double_quant", type=bool, default=True)
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="float16")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4")
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=6)
    parser.add_argument("--saving_checkpoint", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--interp_method", type=str, default="locf")
    parser.add_argument("--min_g", type=float, default=1e-10)
    parser.add_argument(
        "--target_scale",
        type=str,
        default="log",
        choices=["log", "linear"],
        help="Target scale for prediction: log(T) or T.",
    )
    parser.add_argument(
        "--experiment_date",
        type=str,
        default=None,
        help="Fix results folder date (YYMMDD). Defaults to today's date.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.device = torch.device(args.device)
    main(args)
