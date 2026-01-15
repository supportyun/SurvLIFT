import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    get_linear_schedule_with_warmup, 
    set_seed
)
from peft import PeftModel, PeftConfig
from peft import get_peft_model, LoraConfig
from data_utils_share import *
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
import time
from transformers import LogitsProcessor, LogitsProcessorList, TemperatureLogitsWarper
# [수정/추가] data_utils_share에서 새로 만든 Lookup 함수 임포트
from data_utils_share import lookup_time_from_groups, extract_prompts, extract_completion, parse_number_4dec, extract_variable
import pandas as pd # [수정] 로그 저장을 위해 Pandas 임포트

# 여기서 input_ids는 사용자가 처음 입력한 질문, 모델이 방금가지 대답한 내용을 모두 합친 덩어리이다.
class CleanLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids, scores):
        # FP32로 변환 + NaN/Inf 제거 + 과도한 값 클램프
        scores = scores.float() # 다음에 올 단어를 찾는 중에 모든 단어에게 매긴 logit을 의미, bathch size개수 * vocab_size를 의미한다.
        bad = ~torch.isfinite(scores)
        if bad.any():
            scores = scores.clone() # 원본을 복사(clone)한 뒤, 문제가 있는 위치(bad)의 점수를 -1e9(최하점)로 덮어씌워서 선택되지 않게 막는 것.
            scores[bad] = -1e9
        scores = torch.clamp(scores, min=-1e9, max=1e9)
        return scores

class LlamaDataset(Dataset):
    def __init__(self, json_lst, tokenizer, max_length=1024): ## 클래스를 만들때 재료
        # instance 생성할 때 한번만 실행됨. raw text를 읽어서 input_ids, attention mask, labels를 구축하는 중
        #json_lst에는 각각의 줄이 프롬프트, completion이 딕셔너리 형태로 되어 있다. 
        # 이는 prepare_data함수(이 파일에 있는)에서 만들어진다.
        texts = []
        completion_lens = []
        for row in json_lst:
            t = ' '.join(row.values())
            texts.append(t)
            l = len(tokenizer.tokenize(row['completion'])) ###정답 토큰의 개수
            completion_lens.append(l) ## 토큰의 개수를 센다. 나중에 마스킹할 때 쓰려고
        
        #tokens변수 안에는 input_ids, attenton_mask(패딩 위치)가 포함된다.
        tokens = tokenizer(
            texts, 
            truncation=True,  ## texts를 토큰으로 나누는 과정.input_ids(숫자 부분), attention_mask(패딩 여부)
            padding=True, ##가장 긴 문장을 기준
            max_length=max_length, 
            return_tensors='pt'
        )

        
        #input_ids는 tensor다. 데이터 개수 by 시퀀스 길이 형태
        
        self.input_ids = tokens['input_ids'] 
        self.attention_mask = tokens['attention_mask'] ### attetion mask는 패딩인 부분의 가중치를 0으로 만들어준다.
        self.labels = []
        for i in range(len(self.input_ids)):
            b_labels = self.input_ids[i].clone() #i 번째 줄에 대한 것을 가져온다. 문장 단위로 기록.
            b_labels[:-completion_lens[i]] = -100  
            self.labels.append(b_labels) 
        
        #우리의 labels: [-100, -100, -100, 10, ., 40, @@@]
        #앞의 -100 구간(프롬프트): 모델이 뭘 예측하든 채점 안 함. (Loss 0)
        #뒤의 값 있는 구간(정답): 모델이 틀리면 혼냄. (Loss 발생)
        #labels는 train 함수 루프에 나옴.   
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]
    # idx라는 것은 몇 번째 줄 데이터라는 것을 지정해준다.

class LlamaFinetuner:
    def __init__(self,
                 model_name="meta-llama/Llama-3.2-1B",
                 device=torch.device('cuda:0'),
                 output_dir="./results/llama/",
                 load_in_4bit=True,
                 load_in_8bit=False,
                 bnb_4bit_use_double_quant=True,
                 bnb_4bit_compute_dtype="float16",
                 bnb_4bit_quant_type="nf4",
                 r=8,
                 lora_alpha=32,
                 lora_dropout=0.1, 
                 seed=1024):
        
        self.device = device
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device.index)

        set_seed(seed)
        import random                           # 이 부분은 시드 고정으로 randomness 통제하는 부분
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.tokenizer = AutoTokenizer.from_pretrained( 
            model_name, 
            trust_remote_code=True, 
            use_fast=True, 
            token=os.getenv('HF_TOKEN')
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token # llama 모델에서는 tokenizer를 eos token으로 쓴다.
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"  # 학습할 때에는 오른쪽이어도 상관이 없는데 글을 생성할 때에는 왼쪽이어야함.

        self.output_dir = output_dir
        self.base_model_name = model_name
        
        bnb_config = {
            'load_in_4bit': load_in_4bit,   # 32비트 모델을 4비트로 압축해서 로드
            'load_in_8bit': load_in_8bit,
            'bnb_4bit_use_double_quant': bnb_4bit_use_double_quant,
            'bnb_4bit_compute_dtype': bnb_4bit_compute_dtype,   # 계산할 때는 16비트로 (속도)
            'bnb_4bit_quant_type': bnb_4bit_quant_type,
        }

        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,   # 학습 모드
            r=r,    # LoRA 랭크, 크면 파라미터가 많아진다.
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
        # Base Model을 4비트로 로드함, AutoModelForCausalLM은 다음 단어를 예측하는거다.
        self.model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
        # 그 위에 LoRA 어댑터(학습할 얇은 층)을 붙임
        self.model = get_peft_model(self.model, lora_config) #위에거 freeze시키고 얇은 층만 학습시키겠다.
        self.model.to(self.device)
        
    def prepare_data(self, jsonl_path):
        with open(jsonl_path, 'r') as json_file:
            json_lst = [json.loads(line) for line in json_file] # loads가 딕셔너리로 바꿈.
        dataset = LlamaDataset(json_lst, self.tokenizer) ##아까 나온 클래스에다가 넣음. init에 input들어감.
        return dataset
    # 하드디스크에 저장된 텍스트 파일 jsonl을 라인 바이 라인을 읽어가벼 리스트, 딕셔너리로 바꿈.

    def train(self, train_jsonl, val_jsonl, unique_validate_df, # unique_validate_df는 안 쓰지만 호환성 위해 남겨둠
              epochs=3,
              batch_size=4,
              lr=1e-5,
              weight_decay=0.01,
              warmup_steps=6,
              saving_checkpoint=True,
              interp_method="locf", # 안 씀
              min_g = 1e-10,        # 안 씀
              fallback_means=None
              ):
        
        train_dataset = self.prepare_data(train_jsonl)
        val_dataset = self.prepare_data(val_jsonl)
        #####
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) #torch.utils.data 패키지 안에 있는 클래스
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        total_steps = len(train_loader) * epochs
        
        self.train_loss_list, self.val_loss_list = [], []
        val_rmse_list = []  # [수정] C-index 대신 RMSE 저장
        misclassified = []
        misclassified2 = []
        
        # [추가됨] 상세 분석용 리스트 선언
        greedy_success_rate_list = [] # 한 번에 성공한 비율
        retry_success_rate_list = []  # 재시도로 성공한 비율
        fallback_rate_list = []       # 통계적 대치로 메꿔진 비율

        # Zero-shot 평가 (수정된 evaluate_epoch0_model 사용)
        train_loss0, val_loss0, val_rmse0, r_greedy0, r_retry0, r_fallback0, invalid_ratio0, final_invalid_ratio0 = self.evaluate_epoch0_model(
            train_loader, val_loader, val_jsonl, batch_size, unique_validate_df, interp_method, min_g, fallback_means
        )
        
        self.train_loss_list.append(train_loss0)
        self.val_loss_list.append(val_loss0)
        val_rmse_list.append(val_rmse0)
        misclassified.append(invalid_ratio0)
        misclassified2.append(final_invalid_ratio0)
        greedy_success_rate_list.append(r_greedy0)
        retry_success_rate_list.append(r_retry0)
        fallback_rate_list.append(r_fallback0)

        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        
        best_loss = np.inf
        
        for epoch in range(epochs):
            self.model.train()
            tqdm_object = tqdm(train_loader, total=len(train_loader), desc=f"Epoch: {epoch + 1}", dynamic_ncols=True)
            train_loss = []
            for batch in tqdm_object: ## batch의 형태는 길이 3짜리 리스트. 
                self.model.zero_grad() #이전 배치가 다음 배치에 영향 주는 것을 막는다.
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                #### b라는 batch의 각 요소는 각각의 텐서이다. 각 요소가 텐서이다.
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)   
                loss = outputs.loss # 내부적으로 Cross Entropy Loss 계산
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_loss.append(loss.detach().item()) #미분기록까지 있던 loss를 float만 빼서 저장
                tqdm_object.set_postfix(train_loss=np.mean(train_loss))
            
            val_loss = self.validate(val_loader)
            ### 한 에폭 당 데이터 개수/배치 사이즈만큼의 loss개수가 있을 것이다. 그래서 그 개수로 나눠 평균을 내는 것이다.
            self.val_loss_list.append(val_loss)
            self.train_loss_list.append(np.mean(train_loss)) 
            
            # [수정] Validation 평가 데이터 준비
            val_prompts = extract_prompts(val_jsonl, '') 
            val_truth = extract_completion(val_jsonl) 

            # 1. global_mean 설정 및 안전장치
            g_mean = fallback_means.get("global_mean") if fallback_means else None ###없으면 None으로 놓기.
            if g_mean is None:
                g_mean = 5.0 # 수치적 안정성을 위해 0.0 대신 사용
            
            # 2. generate 호출
            # [수정] 로그 반환 요청 (return_logs=True)
            val_generated, invalid_ratio, final_invalid_ratio, val_logs = self.generate(
                text_lst=val_prompts, #####################이게 들어간다는 것
                max_token=10, 
                batch_size=batch_size, 
                valid_mean=g_mean, 
                fallback_means=fallback_means, ##########################
                return_logs=True # [NEW] 상세 로그 받기
            )
            
            # --- [추가됨] 로그 분석 및 비율 계산 ---
            total_samples = len(val_generated)

            # 1. Greedy Success: attempt가 'greedy'이면서 is_valid가 True인 개수
            n_greedy = sum(1 for log in val_logs if log['attempt'] == 'greedy' and log['is_valid'])

            # 2. Retry Success: attempt가 'retry'로 시작하면서 is_valid가 True인 개수
            n_retry = sum(1 for log in val_logs if log['attempt'].startswith('retry') and log['is_valid'])

            # 3. Fallback Stat: attempt가 'fallback_stat'인 개수 (이건 무조건 valid 처리됨)
            n_fallback = sum(1 for log in val_logs if log['attempt'] == 'fallback_stat')

            # 비율 계산
            r_greedy = n_greedy / total_samples
            r_retry = n_retry / total_samples
            r_fallback = n_fallback / total_samples

            # 리스트에 저장
            greedy_success_rate_list.append(r_greedy)
            retry_success_rate_list.append(r_retry)
            fallback_rate_list.append(r_fallback)

            # 문자열 -> 숫자 변환 및 RMSE 계산
            val_preds = np.array(val_generated) 
            val_truth_vals = np.array([float(s.split('@@@', 1)[0].strip()) for s in val_truth])
            
            # RMSE 계산
            mse = np.mean((val_truth_vals - val_preds)**2) 
            val_rmse = np.sqrt(mse) 
            
            print(f"Epoch {epoch+1}: Train Loss: {np.mean(train_loss):.4f}, Val Loss: {val_loss:.4f}, RMSE: {val_rmse:.4f}")
            print(f"   >> [Detail] Greedy: {r_greedy:.1%} | Retry: {r_retry:.1%} | Fallback: {r_fallback:.1%}")

            val_rmse_list.append(val_rmse)
            misclassified.append(invalid_ratio) # 1차 실패율
            misclassified2.append(final_invalid_ratio) # Fallback 적용 후 실패율
            
            # Early Stopping Check
            improved = (val_loss <= best_loss)
            if improved:
                best_loss = val_loss
                es_counter = 0
                print(f"[BEST] val_loss improved. Saving model & logs...")
                if saving_checkpoint:
                    self.save_model(epoch=epoch+1, val_metric=val_rmse, 
                                    val_loss=val_loss, invalid_ratio=invalid_ratio,
                                    final_invalid_ratio=final_invalid_ratio)
                    
                    # [NEW] Best Model의 예측 상세 기록 저장
                    log_df = pd.DataFrame(val_logs)
                    log_path = os.path.join(self.output_dir, "best_model_val_predictions.csv")
                    log_df.to_csv(log_path, index=False, encoding='utf-8-sig')
            else:
                es_counter += 1
                if es_counter >= 5:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # [수정] 메트릭 저장 (RMSE로 변경)
        metrics = {
            'train_loss': self.train_loss_list, # 딕셔너리 형태로 각각의 리스트가 value값으로 들어감.
            'val_loss': self.val_loss_list,
            'val_rmse': val_rmse_list,
            'invalid_answer_ratio': misclassified,
            'final_invalid_answer_ratio': misclassified2,
            'greedy_success_rate': greedy_success_rate_list,
            'retry_success_rate': retry_success_rate_list,
            'fallback_rate': fallback_rate_list
        }

        with open(os.path.join(self.output_dir, "validation_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2) # 딕셔너리리가 f라는 json으로 저장됨
        
        # [수정] 그래프 그리기 (RMSE)
        epochs_range = list(range(len(self.train_loss_list)))
        plot_filename = os.path.join(self.output_dir, f"loss_metric_plot.png")
        fig, ax = plt.subplots(figsize=(8,5)) # 기본적으로 왼쪽 축 y축
        ax2 = ax.twinx() #x축은 그대로 쓰고 y축 그리기.
        #각각은 line을 의미
        l_train, = ax.plot(epochs_range, metrics['train_loss'], marker='o', color='#204e6e', label="Train loss")
        l_val,   = ax.plot(epochs_range, metrics['val_loss'], marker='o', color='#ff7f0e', label="Validation loss")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")

        # RMSE Plot
        m_rmse, = ax2.plot(epochs_range, metrics['val_rmse'], marker='o', color='#d62728', label="Val RMSE")
        ax2.set_ylabel("RMSE")

        lines = [l_train, l_val, m_rmse]
        labels = [h.get_label() for h in lines]
        ax.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
        plt.tight_layout()
        fig.savefig(plot_filename, dpi=300)
        plt.close()

    def validate(self, val_loader):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                losses.append(outputs.loss.item())
        return np.mean(losses)



    def generate(self, text_lst, max_token=10, batch_size=16,
                 valid_temperature=0.75, valid_mean=0.0, stop_str="@@@",
                 retries=5, 
                 clip_min=0.0, clip_max=120.0,
                 fallback_means=None, return_logs=False): # [NEW] return_logs 추가
        
        self.model.eval()
        device = self.device
        results = [None] * len(text_lst)
        all_debug_logs = [] ##모든 대화 기록 (invaid ratio 내용물 확인용)
       
        # ---------------------------------------------------------
        # 1. 1차 시도 (Greedy Search)
        # ---------------------------------------------------------
        for i in range(0, len(text_lst), batch_size):
            texts = text_lst[i:i+batch_size]
            try:
                with torch.no_grad():
                    # toks는 input_id, attention_mask를 모두를 포함하는 딕셔너리.
                    toks = self.tokenizer(texts, truncation=True, padding=True,
                                        max_length=1024, return_tensors='pt').to(device) 
                    
                    # input_ids[0]는 몇개의 문제 = batch size, [1]은 문제의 길이 
                    # 가장 긴 길이 = T_in
                    T_in = toks['input_ids'].shape[1] 
                                                        
                    out_ids = self.model.generate(
                        **toks,
                        max_new_tokens=max_token,
                        do_sample=False, # Greedy
                        pad_token_id=self.tokenizer.eos_token_id, # 어차피 masking 할거야.
                    )
                
                gen_texts = [self.tokenizer.decode(out_ids[b, T_in:], skip_special_tokens=True) for b in range(out_ids.size(0))]
                
                # 한 문장씩 읽으면서 정답 decode하기(파싱)
                for j, txt in enumerate(gen_texts): 
                    val = parse_number_4dec(txt, stop_str=stop_str, clip_min=clip_min, clip_max=clip_max)
                    # [수정] 결과 소수점 2자리 반올림 (parse 함수 내에서 이미 처리되지만 확실히)
                    if val is not None: val = round(val, 2)
                    results[i + j] = val

                    #### [추가 2] Greedy 결과 기록
                    prompt_short = texts[j][-50:] if len(texts[j]) > 50 else texts[j]
                    
                    all_debug_logs.append({
                        "id": i + j,                # 전체 데이터에서의 순서
                        "attempt": "greedy",        # 시도 종류 (greedy / retry_1 / retry_2 ...)
                        "prompt_snippet": prompt_short,
                        "raw_output": txt,          # [핵심] 모델이 뱉은 날것의 문자열
                        "parsed_value": val,        # 파싱된 숫자 (실패시 None)
                        "is_valid": val is not None # 성공 여부
                    })

            except Exception as e:
                print(f"[greedy error] {e}"); time.sleep(2)
        
        # ---------------------------------------------------------
        # 2. 재시도 (Sampling with Retries)
        # ---------------------------------------------------------
        for ret in range(retries):
            # todo_idx가 비어질 때까지 한다. todo_idx = None이 될 때까지
            todo_idx = [k for k, v in enumerate(results) if v is None]
            if not todo_idx: break 
            
            print(f"Retry {ret+1}: {len(todo_idx)} items left")
            
            for s in range(0, len(todo_idx), batch_size):
                idx_chunk = todo_idx[s:s+batch_size]
                
                # None인 부분에 대해 질문을 불러온다.
                texts = [text_lst[k] for k in idx_chunk] 
                
                try:
                    # 학습 중이므로 gradient를 기록하지 않음.
                    with torch.no_grad(): 
                        toks = self.tokenizer(texts, truncation=True, padding=True, max_length=1024, return_tensors='pt').to(device)
                        
                        # toks에서는 None이 나온 해당 질문이 들어있음.
                        T_in = toks['input_ids'].shape[1] 
                        
                        # 1차 시도와 달리 이것을 하는 것은 argmax가 아니라 softmax이기 때문이다.
                        processors = LogitsProcessorList([
                            CleanLogitsProcessor(), 
                            TemperatureLogitsWarper(max(valid_temperature, 1e-5))
                        ])
                        
                        # 정밀하게 float32로 아주 꼼꼼하게 계산하는 것.
                        with torch.amp.autocast(device_type="cuda", enabled=False): 
                            out_ids = self.model.generate(
                                **toks, 
                                max_new_tokens=max_token, 
                                do_sample=True, # greedy 방법이 아니라 sampling을 해서 token을 뱉게 한다.
                                temperature=valid_temperature, 
                                logits_processor=processors, 
                                pad_token_id=self.tokenizer.eos_token_id
                            )
                            
                    gen_texts = [self.tokenizer.decode(out_ids[b, T_in:], skip_special_tokens=True) for b in range(out_ids.size(0))]
                    for j, txt in enumerate(gen_texts):
                        val = parse_number_4dec(txt, stop_str=stop_str, clip_min=clip_min, clip_max=clip_max)
                        if val is not None: 
                            val = round(val, 2) # [수정] 2자리 반올림
                            results[idx_chunk[j]] = val
                        
                        #### [수정됨] 정상 Sampling 시도 결과 기록 (여기 있어야 합니다!)
                        prompt_short = texts[j][-50:] if len(texts[j]) > 50 else texts[j]
                        all_debug_logs.append({
                            "id": idx_chunk[j],
                            "attempt": f"retry_{ret+1}_sampling", # 구분: sampling
                            "prompt_snippet": prompt_short,
                            "raw_output": txt,
                            "parsed_value": val,
                            "is_valid": val is not None
                        })
                
                # try를 하다가 run time error가 발생하면 (Sampling 실패) Greedy 방법으로 재시도
                except RuntimeError as e: 
                    print(f"[sampling error -> greedy fallback] {e}")
                    
                    # [Nested Try] 안전 모드(Greedy)로 재시도
                    try:
                        # 보통은 autocast = True로 사용해서 float16을 섞는데 이거 끄기. 꼼꼼하게 계산하라는 뜻.
                        with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=False): 
                            out_ids = self.model.generate(**toks, max_new_tokens=max_token, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
                        
                        gen_texts = [self.tokenizer.decode(out_ids[b, T_in:], skip_special_tokens=True) for b in range(out_ids.size(0))]
                        for j, txt in enumerate(gen_texts):
                            val = parse_number_4dec(txt, stop_str=stop_str, clip_min=clip_min, clip_max=clip_max)
                            if val is not None: results[idx_chunk[j]] = val

                        #### [수정됨] Fallback Greedy 시도 결과 기록
                            prompt_short = texts[j][-50:] if len(texts[j]) > 50 else texts[j]
                            all_debug_logs.append({
                                "id": idx_chunk[j],
                                "attempt": f"retry_{ret+1}_fallback_greedy", # 구분: fallback인 경우에는 이런 게 붙음.
                                "prompt_snippet": prompt_short,
                                "raw_output": txt,
                                "parsed_value": val,
                                "is_valid": val is not None
                            })
                    
                    # ee는 에러 정보를 담고 있는 객체이다.
                    except Exception as ee:
                        print(f"[greedy fallback error] {ee}") 

        # ---------------------------------------------------------
        # 3. 실패 현황 파악 (Diagnosis)
        # ---------------------------------------------------------
        # results에서 None(실패)인 인덱스만 골라냄
        failed_idx = [k for k, v in enumerate(results) if v is None]
        
        # [수정 완료] 위에서 구한 인덱스(k)를 이용해 '실패한 질문 텍스트'를 미리 추출
        failed_prompts = [text_lst[k] for k in failed_idx]

        # 실패한 질문들에서 환자 정보(나이, 치료법 등)를 추출 (Fallback에서 쓰기 위해)
        parsed = extract_variable(failed_prompts)   
        
        # 1차 실패율 계산 (Fallback 전)
        invalid_ratio = len(failed_idx) / len(text_lst)
        print(f"[generate] Model failed ratio (before fallback): {invalid_ratio:.4f}")

        # ---------------------------------------------------------
        # 4. 통계적 대치법 적용 (Statistical Fallback)
        # ---------------------------------------------------------
        # 실패한 게 있고(failed_idx), 족보(fallback_means)도 있다면 시도
        if failed_idx and fallback_means is not None:
            
            for off, k in enumerate(failed_idx):
                if results[k] is not None: continue
                
                # [전략 1] 전체 평균(valid_mean)
                filled = float(valid_mean) 
                
                try:
                    # [전략 2] 그룹 평균 조회
                    age_i, trt_i = parsed["age"][off], parsed["trt"][off]
                    
                    if age_i is not None and trt_i is not None:
                        cand = lookup_time_from_groups(age=age_i, trt=trt_i, means_dict=fallback_means)
                        if cand is not None: filled = float(cand)
                
                except Exception as e:
                    print(f"[fallback lookup error idx={k}] {e}")
                
                # [수정] Fallback 값도 소수점 2자리 반올림
                filled = round(filled, 2)
                results[k] = float(filled)

                ### [추가 4-옵션] Fallback으로 채워진 사실 기록 
                all_debug_logs.append({
                    "id": k,
                    "attempt": "fallback_stat",
                    "prompt_snippet": "STATISTICAL FILL",
                    "raw_output": "None",
                    "parsed_value": filled,
                    "is_valid": True
                })
        # ---------------------------------------------------------
        # 5. 최종 안전장치 및 반환 (Final Cleanup)
        # ---------------------------------------------------------
        # 혹시라도 여전히 None이거나 NaN이 남아있다면 무조건 valid_mean(여기서는 전체 평균)으로 채움
        valid_mean_rounded = round(float(valid_mean), 2)
        results = [valid_mean_rounded if (v is None or v != v) else float(v) for v in results]
        
        final_invalid_ratio = sum(1 for v in results if v is None) / len(text_lst) 
        
        #### [추가 5] CSV 파일로 저장
        # [수정] return_logs 플래그에 따라 반환 값 조정
        if return_logs:
            return results, invalid_ratio, final_invalid_ratio, all_debug_logs
        else:
            # 기존 호환성 유지
            return results, invalid_ratio, final_invalid_ratio

        
    # 업데이트 없이 순수하게 처음 세팅일 때 loss가 얼마인지 계산해서 baseline으로 섦정.
    def evaluate_epoch0_model(self, train_loader, val_loader, val_jsonl, batch_size, unique_validate_df, interp_method, min_g, fallback_means):
        self.model.eval()
        tqdm_object = tqdm(train_loader, total=len(train_loader), desc=f"Epoch: 0", dynamic_ncols=True)
        tr_loss = []
        
        with torch.no_grad():
            for batch in tqdm_object:
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                tr_loss.append(loss.detach().item()) 
                tqdm_object.set_postfix(train_loss=np.mean(tr_loss)) #trainloss를 오른쪽에 딕셔너리 형태로 업데이트해서 이를 표시.
        
        val_loss = self.validate(val_loader)
        print(f"Epoch 0: Train Loss: {np.mean(tr_loss):.4f}, Val Loss: {val_loss:.4f}")
        
        val_prompts = extract_prompts(val_jsonl, '') #질문뒤에 아무것도 안붙여서 extract
        val_truth = extract_completion(val_jsonl)

        current_valid_mean = 5.0
        if fallback_means is not None and "global_mean" in fallback_means:
            current_valid_mean = fallback_means["global_mean"]

        val_generated, invalid_ratio, final_invalid_ratio, val_logs = self.generate(
            text_lst=val_prompts, 
            max_token=10, 
            batch_size=batch_size, 
            valid_mean=current_valid_mean, 
            fallback_means=fallback_means,
            return_logs=True
        )
        
        # [추가] 실제 비율 계산 로직 (train 루프와 동일)
        total_samples = len(val_generated)
        n_greedy = sum(1 for log in val_logs if log['attempt'] == 'greedy' and log['is_valid'])
        n_retry = sum(1 for log in val_logs if log['attempt'].startswith('retry') and log['is_valid'])
        n_fallback = sum(1 for log in val_logs if log['attempt'] == 'fallback_stat')

        r_greedy = n_greedy / total_samples
        r_retry = n_retry / total_samples
        r_fallback = n_fallback / total_samples

        val_preds = np.array(val_generated)
        val_truth_vals = np.array([float(s.split('@@@', 1)[0].strip()) for s in val_truth])
        val_rmse = np.sqrt(np.mean((val_truth_vals - val_preds)**2))
        
        return np.mean(tr_loss), val_loss, val_rmse, r_greedy, r_retry, r_fallback, invalid_ratio, final_invalid_ratio

    def save_model(self, epoch=None, val_metric=None, val_loss=None, invalid_ratio=None, final_invalid_ratio=None):
        os.makedirs(self.output_dir, exist_ok=True)
        print("Saving model to %s" % self.output_dir)
        self.model.save_pretrained(self.output_dir) # self.output_dir에 model, tokenizer을 저장.
        self.tokenizer.save_pretrained(self.output_dir)

        if epoch is not None and val_metric is not None:
            info = {
                "epoch": epoch,
                "val_loss": val_loss,
                "val_rmse": val_metric,
                "invalid_ratio": invalid_ratio,
                "final_invalid_ratio": final_invalid_ratio
            }
            with open(os.path.join(self.output_dir, "best_model_info.json"), "w") as f:
                json.dump(info, f, indent=2)

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.output_dir)
        base_model = AutoModelForCausalLM.from_pretrained(self.output_dir)
        self.model = PeftModel.from_pretrained(base_model, self.output_dir)
        self.model.to(self.device)

    def load_model2(self):
        import os, json, torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        try:
            from transformers import BitsAndBytesConfig
            has_bnb = True
        except Exception:
            has_bnb = False
        from peft import PeftModel

        base_name = getattr(self, "base_model_name", None)
        if base_name is None:
            cfg_path = os.path.join(self.output_dir, "adapter_config.json")
            if os.path.exists(cfg_path):
                try:
                    with open(cfg_path, "r") as f:
                        base_name = json.load(f).get("base_model_name_or_path")
                except Exception:
                    pass
        if base_name is None:
            base_name = "meta-llama/Llama-3.2-1B"

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.output_dir, use_fast=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(base_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        load_in_4bit  = getattr(self, "load_in_4bit",  True)
        load_in_8bit  = getattr(self, "load_in_8bit",  False)
        use_double_q  = getattr(self, "bnb_4bit_use_double_quant", True)
        quant_type    = getattr(self, "bnb_4bit_quant_type", "nf4")
        compute_dtype = getattr(self, "bnb_4bit_compute_dtype", "float16")
        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32,
                    torch.float16: torch.float16, torch.bfloat16: torch.bfloat16, torch.float32: torch.float32}
        compute_dtype = dtype_map.get(compute_dtype, torch.float16)

        if has_bnb:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=(False if load_in_4bit else load_in_8bit),
                bnb_4bit_use_double_quant=use_double_q,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=quant_type,
            )
            base = AutoModelForCausalLM.from_pretrained(
                base_name, trust_remote_code=True, quantization_config=bnb_cfg
            )
        else:
            base = AutoModelForCausalLM.from_pretrained(base_name, trust_remote_code=True)

        self.model = PeftModel.from_pretrained(base, self.output_dir)
        self.model.to(self.device).eval()
