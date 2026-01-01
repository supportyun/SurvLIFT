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

class CleanLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids, scores):
        # FP32로 변환 + NaN/Inf 제거 + 과도한 값 클램프
        scores = scores.float()
        bad = ~torch.isfinite(scores)
        if bad.any():
            scores = scores.clone()
            scores[bad] = -1e9
        scores = torch.clamp(scores, min=-1e9, max=1e9)
        return scores

class LlamaDataset(Dataset):
    def __init__(self, json_lst, tokenizer, max_length=1024):
        texts = []
        completion_lens = []
        for row in json_lst:
            t = ' '.join(row.values())
            texts.append(t)
            l = len(tokenizer.tokenize(row['completion']))
            completion_lens.append(l)
        
        tokens = tokenizer(
            texts, 
            truncation=True, 
            padding=True, 
            max_length=max_length, 
            return_tensors='pt'
        )

        self.input_ids = tokens['input_ids'] 
        self.attention_mask = tokens['attention_mask'] 
        self.labels = []
        for i in range(len(self.input_ids)):
            b_labels = self.input_ids[i].clone()
            b_labels[:-completion_lens[i]] = -100  
            self.labels.append(b_labels)
            
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]

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
        import random
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
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        self.output_dir = output_dir
        self.base_model_name = model_name
        
        bnb_config = {
            'load_in_4bit': load_in_4bit,
            'load_in_8bit': load_in_8bit,
            'bnb_4bit_use_double_quant': bnb_4bit_use_double_quant,
            'bnb_4bit_compute_dtype': bnb_4bit_compute_dtype,
            'bnb_4bit_quant_type': bnb_4bit_quant_type,
        }

        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
        self.model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
        self.model = get_peft_model(self.model, lora_config) 
        self.model.to(self.device)
        
    def prepare_data(self, jsonl_path):
        with open(jsonl_path, 'r') as json_file:
            json_lst = [json.loads(line) for line in json_file]
        dataset = LlamaDataset(json_lst, self.tokenizer)
        return dataset

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
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        total_steps = len(train_loader) * epochs
        
        self.train_loss_list, self.val_loss_list = [], []
        val_rmse_list = []  # [수정] C-index 대신 RMSE 저장
        misclassified = []
        misclassified2 = []
        
        # Zero-shot 평가 (수정된 evaluate_epoch0_model 사용)########
        train_loss0, val_loss0, val_rmse0, _, invalid_ratio0, final_invalid_ratio0 = self.evaluate_epoch0_model(
            train_loader, val_loader, val_jsonl, batch_size, unique_validate_df, interp_method, min_g, fallback_means
        )
        
        self.train_loss_list.append(train_loss0)
        self.val_loss_list.append(val_loss0)
        val_rmse_list.append(val_rmse0)
        misclassified.append(invalid_ratio0)
        misclassified2.append(final_invalid_ratio0)

        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        
        best_loss = np.inf
        
        for epoch in range(epochs):
            self.model.train()
            tqdm_object = tqdm(train_loader, total=len(train_loader), desc=f"Epoch: {epoch + 1}", dynamic_ncols=True)
            train_loss = []
            for batch in tqdm_object:
                self.model.zero_grad()
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)   
                loss = outputs.loss # 내부적으로 Cross Entropy Loss 계산
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_loss.append(loss.detach().item())
                tqdm_object.set_postfix(train_loss=np.mean(train_loss))
            
            val_loss = self.validate(val_loader)
            self.train_loss_list.append(np.mean(train_loss))
            self.val_loss_list.append(val_loss)
            print(f"Epoch {epoch+1}: Train Loss: {np.mean(train_loss):.4f}, Val Loss: {val_loss:.4f}")

            # [수정] Validation 평가 데이터 준비
            val_prompts = extract_prompts(val_jsonl, '') 
            val_truth = extract_completion(val_jsonl) 

            # 1. global_mean 설정 및 안전장치
            g_mean = fallback_means.get("global_mean") if fallback_means else None
            if g_mean is None:
                print("Warning: global_mean not found in fallback_means. Using default 5.0")
                g_mean = 5.0 # 수치적 안정성을 위해 0.0 대신 사용
            
            # 2. generate 호출
            val_generated, invalid_ratio, final_invalid_ratio = self.generate(
                text_lst=val_prompts, 
                max_token=10, 
                batch_size=batch_size, 
                valid_mean=g_mean, 
                fallback_means=fallback_means
            )
            
            # 문자열 -> 숫자 변환 및 RMSE 계산
            val_preds = np.array(val_generated) 
            val_truth_vals = np.array([float(s.split('@@@', 1)[0].strip()) for s in val_truth])
            
            # RMSE 계산
            mse = np.mean((val_truth_vals - val_preds)**2) 
            val_rmse = np.sqrt(mse) 
            
            print(f"Epoch {epoch+1} RMSE: {val_rmse:.4f}, Invalid Ratio: {invalid_ratio:.4f}")

            val_rmse_list.append(val_rmse)
            misclassified.append(invalid_ratio) # 1차 실패율
            misclassified2.append(final_invalid_ratio) # Fallback 적용 후 실패율
            
            # Early Stopping Check
            improved = (val_loss <= best_loss)
            if improved:
                best_loss = val_loss
                es_counter = 0
                print(f"[BEST] val_loss improved. Saving model...")
                if saving_checkpoint:
                    self.save_model(epoch=epoch+1, val_metric=val_rmse, 
                                    val_loss=val_loss, invalid_ratio=invalid_ratio,
                                    final_invalid_ratio=final_invalid_ratio)
            else:
                es_counter += 1
                if es_counter >= 5:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # [수정] 메트릭 저장 (RMSE로 변경)
        metrics = {
            'train_loss': self.train_loss_list,
            'val_loss': self.val_loss_list,
            'val_rmse': val_rmse_list,
            'invalid_answer_ratio': misclassified,
            'final_invalid_answer_ratio': misclassified2
        }

        with open(os.path.join(self.output_dir, "validation_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        
        # [수정] 그래프 그리기 (RMSE)
        epochs_range = list(range(len(self.train_loss_list)))
        plot_filename = os.path.join(self.output_dir, f"loss_metric_plot.png")
        fig, ax = plt.subplots(figsize=(8,5))
        ax2 = ax.twinx()

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
                 retries=5, clip_min=0.0, clip_max=None, 
                 fallback_means=None):
        self.model.eval()
        device = self.device
        results = [None] * len(text_lst)

        # 1. 1차 시도 (Greedy Search)
        for i in range(0, len(text_lst), batch_size):
            texts = text_lst[i:i+batch_size]
            try:
                with torch.no_grad():
                    toks = self.tokenizer(texts, truncation=True, padding=True,
                                        max_length=1024, return_tensors='pt').to(device)
                    T_in = toks['input_ids'].shape[1] 
                    out_ids = self.model.generate(
                        **toks,
                        max_new_tokens=max_token,
                        do_sample=False, # Greedy
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                gen_texts = [self.tokenizer.decode(out_ids[b, T_in:], skip_special_tokens=True) for b in range(out_ids.size(0))]
                for j, txt in enumerate(gen_texts):
                    results[i + j] = parse_number_4dec(txt, stop_str=stop_str, clip_min=clip_min, clip_max=clip_max)
            except Exception as e:
                print(f"[greedy error] {e}"); time.sleep(2)
        
        # 2. 재시도 (Sampling with Retries)
        for ret in range(retries):
            todo_idx = [k for k, v in enumerate(results) if v is None]
            if not todo_idx: break
            print(f"Retry {ret+1}: {len(todo_idx)} items left")
            
            for s in range(0, len(todo_idx), batch_size):
                idx_chunk = todo_idx[s:s+batch_size]
                texts = [text_lst[k] for k in idx_chunk]
                try:
                    with torch.no_grad():
                        toks = self.tokenizer(texts, truncation=True, padding=True, max_length=1024, return_tensors='pt').to(device)
                        T_in = toks['input_ids'].shape[1]
                        processors = LogitsProcessorList([
                            CleanLogitsProcessor(), 
                            TemperatureLogitsWarper(max(valid_temperature, 1e-5))
                        ])
                        with torch.amp.autocast(device_type="cuda", enabled=False):
                            out_ids = self.model.generate(
                                **toks, 
                                max_new_tokens=max_token, 
                                do_sample=True, 
                                temperature=valid_temperature, 
                                logits_processor=processors, 
                                pad_token_id=self.tokenizer.eos_token_id
                            )
                    gen_texts = [self.tokenizer.decode(out_ids[b, T_in:], skip_special_tokens=True) for b in range(out_ids.size(0))]
                    for j, txt in enumerate(gen_texts):
                        val = parse_number_4dec(txt, stop_str=stop_str, clip_min=clip_min, clip_max=clip_max)
                        if val is not None: results[idx_chunk[j]] = val
                
                except RuntimeError as e:
                    print(f"[sampling error -> greedy fallback] {e}")
                    try:
                        with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=False):
                            out_ids = self.model.generate(**toks, max_new_tokens=max_token, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
                        gen_texts = [self.tokenizer.decode(out_ids[b, T_in:], skip_special_tokens=True) for b in range(out_ids.size(0))]
                        for j, txt in enumerate(gen_texts):
                            val = parse_number_4dec(txt, stop_str=stop_str, clip_min=clip_min, clip_max=clip_max)
                            if val is not None: results[idx_chunk[j]] = val
                    except Exception as ee:
                        print(f"[greedy fallback error] {ee}")

        # 실패율 계산
        failed_idx = [k for k, v in enumerate(results) if v is None]
        invalid_ratio = len(failed_idx) / len(text_lst)
        print(f"[generate] Model failed ratio (before fallback): {invalid_ratio:.4f}")

        # 3. 최종 안전장치 (Fallback with Statistical Means)
        if failed_idx and fallback_means is not None:
            failed_prompts = [text_lst[k] for k in failed_idx]
            parsed = extract_variable(failed_prompts)
            
            for off, k in enumerate(failed_idx):
                if results[k] is not None: continue
                
                filled = float(valid_mean) 
                try:
                    age_i, trt_i = parsed["age"][off], parsed["trt"][off]
                    if age_i is not None and trt_i is not None:
                        cand = lookup_time_from_groups(age=age_i, trt=trt_i, means_dict=fallback_means)
                        if cand is not None: filled = float(cand)
                except Exception as e:
                    print(f"[fallback lookup error idx={k}] {e}")
                
                results[k] = float(filled)

        results = [float(valid_mean) if (v is None or v != v) else float(v) for v in results]
        final_invalid_ratio = sum(1 for v in results if v is None) / len(text_lst) #0이어야함
        return results, invalid_ratio, final_invalid_ratio

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
                tqdm_object.set_postfix(train_loss=np.mean(tr_loss))
        
        val_loss = self.validate(val_loader)
        print(f"Epoch 0: Train Loss: {np.mean(tr_loss):.4f}, Val Loss: {val_loss:.4f}")
        
        val_prompts = extract_prompts(val_jsonl, '')
        val_truth = extract_completion(val_jsonl)

        current_valid_mean = 5.0
        if fallback_means is not None and "global_mean" in fallback_means:
            current_valid_mean = fallback_means["global_mean"]

        val_generated, invalid_ratio, final_invalid_ratio = self.generate(
            text_lst=val_prompts, 
            max_token=10, 
            batch_size=batch_size, 
            valid_mean=current_valid_mean, 
            fallback_means=fallback_means
        )
        
        val_preds = np.array(val_generated)
        val_truth_vals = np.array([float(s.split('@@@', 1)[0].strip()) for s in val_truth])
        val_rmse = np.sqrt(np.mean((val_truth_vals - val_preds)**2))
        
        return np.mean(tr_loss), val_loss, val_rmse, 0.0, invalid_ratio, final_invalid_ratio

    def save_model(self, epoch=None, val_metric=None, val_loss=None, invalid_ratio=None, final_invalid_ratio=None):
        os.makedirs(self.output_dir, exist_ok=True)
        print("Saving model to %s" % self.output_dir)
        self.model.save_pretrained(self.output_dir)
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