import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import argparse
import torch
import pandas as pd
import numpy as np
from huggingface_hub import login
import os
login(token=os.getenv("HF_TOKEN"))
from llama_finetuner_logit_cp_share import LlamaFinetuner
from sklearn.model_selection import train_test_split
from data_utils_share import * ####전부 import하는 것
from transformers import set_seed
from datetime import datetime  # 

def prepare_data(data_seed):
    # [수정 후] 현재 파일 위치를 기준으로 한 '절대 경로' 사용 (가장 안전함)
    # 1. 현재 이 스크립트(main_logit...py)가 있는 폴더 경로 (regression 폴더)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. 그 상위 폴더 (SurvLIFT 프로젝트 루트)
    project_root = os.path.dirname(script_dir)
    
    # 3. 데이터 파일 경로 결합 (SurvLIFT/data/target_data_for_aft_seed1.csv)
    file_path = os.path.join(project_root, "data", f"target_data_for_aft_seed{data_seed}.csv")
    
    print(f"Loading data from: {file_path}") # 경로 확인용 로그 출력

    if not os.path.exists(file_path):
        # 파일이 진짜 없는 경우를 대비해 현재 data 폴더에 뭐가 있는지 보여줌
        data_dir = os.path.dirname(file_path)
        if os.path.exists(data_dir):
            print(f"Files in data dir: {os.listdir(data_dir)}")
        else:
            print(f"Data directory does not exist: {data_dir}")
            
        raise FileNotFoundError(f"Data file not found: {file_path}")
        
    df = pd.read_csv(file_path)
    
    # Train/Val/Test Split (기존 로직 유지)
    event_data = df[df['status'] == 1]
    censor_data = df[df['status'] == 0]
    # train:validation:test = 8:1:1로 만드는 과정.
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
# args 는 터미널에 입력한 값들을 객체 형태로 저장, 실험 설정값들의 모음집이라고 보면 됨. 예) arg.epochs
def main(args):
    print(f"Prepare data (Seed: {args.data_seed})...")
    set_seed(args.seed) ##huggingface transformer 라이브러리 함수
    
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
    #
    print("Save prompts...")

    # 1. [필수] 오늘 날짜와 기본 경로(base_dir)를 먼저 정의해야 합니다!
    today_date = datetime.now().strftime("%y%m%d")
    
    base_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        f"results/{today_date}/data{args.data_seed}/seed_{args.seed}"
    )
    # [수정 후] 실험 세팅(Rank, Alpha, LR, Dropout)이 모두 폴더명에 들어가도록 변경
    folder_name = (
        f"epochs_{args.epochs}_"
        f"rank{args.r}_"       # args.r 확인 (parser에 정의된 이름)
        f"alpha{args.lora_alpha}_"
        f"lr{args.lr}_"
        f"drop{args.lora_dropout}"
    )
    
    output_dir = os.path.join(base_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)

    train_js_path = os.path.join(output_dir, "synthetic_prompts_train.jsonl")
    val_js_path   = os.path.join(output_dir, "synthetic_prompts_val.jsonl")
    test_js_path  = os.path.join(output_dir, "synthetic_prompts_test.jsonl")

    #딕셔너리가 리스트 안에 배열되어 있는데 이거를 \n을 기준으로 jsonl으로 합침
    write_jsonl('\n'.join(train_prompts), train_js_path)
    write_jsonl('\n'.join(val_prompts), val_js_path)
    write_jsonl('\n'.join(test_prompts), test_js_path)
    
    # [수정 후] 
    # 1. Train 데이터를 사용해 그룹별 평균(족보) 생성
    # (주의: make_target.py를 통해 만들어진 csv에는 'target_time' 컬럼이 있어야 합니다)
    print("Building group means for fallback...")
    
    # data_utils_share.py에 추가한 build_time_group_means 함수 사용
    group_means_dict = build_time_group_means(train, target_col="target_time")
    
    # 2. 전체 평균도 미리 계산 (Global Mean) - 최후의 보루
    # [수정] Train Mean도 소수점 2자리 반올림
    train_mean = round(train['target_time'].mean(), 2)
    print(f"Train Global Mean: {train_mean:.2f}") 
    
    print('Start train...')                 ##클래스 불러서 객체 만들기
    finetuner = LlamaFinetuner(
        model_name=args.model_name,
        device=args.device,
        output_dir=output_dir,
        load_in_4bit=args.load_in_4bit, #4비트로 압축해서 GPU메모리 아끼기
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant, #이중 양자화
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype, #비트
        bnb_4bit_quant_type=args.bnb_4bit_quant_type, #압축 방식
        r=args.r, #LoRA를 사용하는데, rank를 의미
        lora_alpha=args.lora_alpha, # Scaling Factor(학습 반영 비율)
        lora_dropout=args.lora_dropout,
        seed = args.seed
    )

    # 학습 (unique_validate_df 등 불필요한 인자는 제거하거나 None 처리)
    finetuner.train(train_js_path, val_js_path,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    weight_decay=args.weight_decay, #overfitting 방지
                    warmup_steps=args.warmup_steps,
                    saving_checkpoint=args.saving_checkpoint,
                    unique_validate_df = None,  # 시간 예측에서는 불필요
                    interp_method = args.interp_method,
                    min_g = args.min_g,
                    fallback_means = group_means_dict
                    )
    ####Validation 관련 내용은 llama finetuner 파일에 있음.
    # --- 평가 (Evaluate) ---
    print("Start evaluate...")
    # finetuner.load_model()
    # [수정 후] 더 강력하고 안전한 로딩 함수 사용
    finetuner.load_model2()
    
    test_prompts_loaded = extract_prompts(test_js_path, '')
    test_completions = extract_completion(test_js_path)
    
    # 정답 추출
    test_truth = [float(s.split('@@@', 1)[0].strip()) for s in test_completions]
    
    # 예측
    # ans, invalid_ratio, final_invalid_ratio
    #  = finetuner.generate(text_lst=test_prompts, max_token=10,
    #  batch_size=args.batch_size, valid_mean = 0.0) -> hazard 0 대신에 그냥 평균 수명으로...
    # [수정 후]
    ans, invalid_ratio, final_invalid_ratio, test_logs = finetuner.generate( ####이 부분에서
        text_lst=test_prompts_loaded, 
        max_token=10, 
        batch_size=args.batch_size, 
        
        # 1. 최후의 보루: Train 데이터의 전체 평균 사용 
        valid_mean=train_mean, 
        
        # 2. 1차 방어막: Train 데이터로 만든 그룹별 평균 족보 사용
        fallback_means=group_means_dict,
        
        # [수정] Linear Scale 고려한 범위 설정 (Clip)
        clip_min=0.0,
        clip_max=120.0,
        
        # [NEW] Test 상세 로그 받기
        return_logs=True
    )

    # [NEW] Test 상세 로그 파일 저장
    log_df = pd.DataFrame(test_logs)
    log_path = os.path.join(output_dir, "test_predictions_detailed_logs.csv")
    log_df.to_csv(log_path, index=False, encoding='utf-8-sig')
    print(f"Saved test logs to {log_path}")

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
    
    # 헤더가 없으면 새로 만듦 (Rank 컬럼 추가)
    if not os.path.exists(summary_file):
        with open(summary_file, "w") as f:
            f.write("Data_Seed,Model_Seed,Rank,LR,RMSE,Invalid_Ratio\n")
            
    # 내용 추가 (Rank, LR 정보 포함)
    with open(summary_file, "a") as f:
        f.write(f"{args.data_seed},{args.seed},{args.r},{args.lr},{rmse:.4f},{final_invalid_ratio:.4f}\n")
    print(f"Saved summary to {summary_file}")

    # [수정] 굳이 함수 만들지 않고 바로 딕셔너리 만들어서 저장
    metrics = {
        "test_rmse": rmse,
        "invalid_answer_ratio": invalid_ratio,
        "final_invalid_answer_ratio": final_invalid_ratio   ####invalid ratio의 차이
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
    
    main(args) ### args는 설정값들의 모음
