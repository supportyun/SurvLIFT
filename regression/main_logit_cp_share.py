import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import argparse
import torch
import pandas as pd
import numpy as np
from huggingface_hub import login  # [추가]
login(token="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")
from llama_finetuner_logit_cp_share import LlamaFinetuner
from sklearn.model_selection import train_test_split
from data_utils_share import *
from transformers import set_seed

def prepare_data(data_seed):
    # 상대 경로로 타겟 데이터 로드
    file_path = f"../data/target_data_for_aft_seed{data_seed}.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
        
    df = pd.read_csv(file_path)
    
    # Train/Val/Test Split (기존 로직 유지)
    event_data = df[df['status'] == 1]
    censor_data = df[df['status'] == 0]

    event_train, event_temp = train_test_split(event_data, test_size=0.2, random_state=42)
    event_validate, event_test = train_test_split(event_temp, test_size=0.5, random_state=42)
    censor_train, censor_temp = train_test_split(censor_data, test_size=0.2, random_state=42)
    censor_validate, censor_test = train_test_split(censor_temp, test_size=0.5, random_state=42)
    # frac = 1.0 전체 데이터를 뽑는 것이므로 순서만 바뀜, random_stat은 seed랑 같음
    train = pd.concat([event_train, censor_train]).sample(frac=1, random_state=42)
    validate = pd.concat([event_validate, censor_validate]).sample(frac=1, random_state=42)
    test = pd.concat([event_test, censor_test]).sample(frac=1, random_state=42)

    return train, validate, test

### 프롬프트 변환, 변환후 jsonl 파일로 저장
# args 는 터미널에 입력한 값들을 객체 형태로 저장, 실험 설정값들의 모음집이라고 보면 됨.
def main(args):
    print(f"Prepare data (Seed: {args.data_seed})...")
    set_seed(args.seed)
    
    # 1. 데이터 로드
    train, validate, test = prepare_data(args.data_seed)
    
    # 2. [중요] 불필요한 전처리(Counting process) 삭제 
    # target_time이나 log_time 등 필요한 컬럼이 이미 있는지 확인 필요
    
    init= ''
    end = ''

    # 3. 프롬프트 변환 (data_utils_share.py의 data2text_cp2가 수정되어 있어야 함)
    train_prompts = df2prompts(train, data2text_cp2, init, end)
    val_prompts = df2prompts(validate, data2text_cp2, init, end)
    test_prompts = df2prompts(test, data2text_cp2, init, end)

    print("Save prompts...")
    base_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        f"results/250923/data{args.data_seed}/seed_{args.seed}"
    )    
    output_dir = os.path.join(base_dir, f"epochs_{args.epochs}_lr_{args.lr}")
    os.makedirs(output_dir, exist_ok=True)

    train_js_path = os.path.join(output_dir, "synthetic_prompts_train.jsonl")
    val_js_path   = os.path.join(output_dir, "synthetic_prompts_val.jsonl")
    test_js_path  = os.path.join(output_dir, "synthetic_prompts_test.jsonl")

    write_jsonl('\n'.join(train_prompts), train_js_path)
    write_jsonl('\n'.join(val_prompts), val_js_path)
    write_jsonl('\n'.join(test_prompts), test_js_path)
    #fallback_means 처리 (None)
    # 원본: build_group_means(...) 함수로 그룹별 평균 Hazard를 미리 계산해 뒀습니다. (모델이 답을 못 낼 경우 대타로 쓰기 위해)
    # Fallback means는 필요 시 구현 (여기서는 생략 가능하거나 train 데이터 기반으로 계산)
    fallback_means = None 
    
    print('Start train...')
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
        seed = args.seed
    )

    # 학습 (unique_validate_df 등 불필요한 인자는 제거하거나 None 처리)
    finetuner.train(train_js_path, val_js_path,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    warmup_steps=args.warmup_steps,
                    saving_checkpoint=args.saving_checkpoint,
                    unique_validate_df = None,  # 시간 예측에서는 불필요
                    interp_method = args.interp_method,
                    min_g = args.min_g,
                    fallback_means = fallback_means
                    )
    
    # --- 평가 (Evaluate) ---
    print("Start evaluate...")
    finetuner.load_model()
    
    test_prompts_loaded = extract_prompts(test_js_path, '')
    test_completions = extract_completion(test_js_path)
    
    # 정답 추출
    test_truth = [float(s.split('@@@', 1)[0].strip()) for s in test_completions]
    
    # 예측
    ans, invalid_ratio, final_invalid_ratio = finetuner.generate(
        text_lst=test_prompts_loaded, 
        max_token=10, 
        batch_size=args.batch_size, 
        valid_mean=np.mean(test_truth)
    )

    print("Predictions sample:", ans[:5])

    # [수정 1] MAE, C-index 삭제하고 RMSE 계산으로 변경
    # RMSE = sqrt(mean((True - Pred)^2))
    mse = np.mean((np.array(test_truth) - np.array(ans))**2)
    rmse = np.sqrt(mse)
    
    print(f"Test RMSE: {rmse:.4f}")

    # [수정 2] 결과 상세 저장 (C-index 관련 내용 제거)
    result_df = pd.DataFrame({
        'id': test['id'].values,
        'true_target_time': test_truth,
        'pred_time': ans,
        'observed_time': test['time'].values,
        'status': test['status'].values
    })
    result_df.to_csv(os.path.join(output_dir, "prediction_results.csv"), index=False)

    # [수정 3] 요약 결과 저장 (CSV 헤더 및 내용 변경)
    summary_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_summary_results.csv")
    
    if not os.path.exists(summary_file):
        with open(summary_file, "w") as f:
            # 헤더에서 MAE, C_index 제거하고 RMSE 추가
            f.write("Data_Seed,Model_Seed,RMSE,Invalid_Ratio\n")
            
    with open(summary_file, "a") as f:
        # 내용도 RMSE로 변경
        f.write(f"{args.data_seed},{args.seed},{rmse:.4f},{final_invalid_ratio:.4f}\n")
        
    print(f"Saved summary to {summary_file}")

    # [수정] 굳이 함수 만들지 않고 바로 딕셔너리 만들어서 저장
    metrics = {
        "test_rmse": rmse,
        "invalid_answer_ratio": invalid_ratio,
        "final_invalid_answer_ratio": final_invalid_ratio
    }
    
    # test_metrics.json 파일로 저장
    json_path = os.path.join(output_dir, "test_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("All evaluation finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Args 설정 (중복 제거 및 정리)
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
    parser.add_argument("--seed", type=int, default=1024) # 모델 시드
    parser.add_argument("--interp_method", type=str, default="locf")
    parser.add_argument("--min_g", type=float, default=1e-10)
    
    args = parser.parse_args()
    args.device = torch.device(args.device)
    
    main(args)