#!/bin/bash
set -e
cd "$(dirname "$0")"

# Seed i만 실행
for i in 1
do
    echo "🚀 Start Training Seed $i (Standard 8B Config)"

    
    # [중요] r=16, alpha=32 (똑똑한 모델용 표준 설정)
    # [중요] lr=1e-4 (안정적 학습)
    
    HF_TOKEN="" nohup python main_logit_cp_share.py \
      --model_name "meta-llama/Llama-3.1-8B" \
      --data_seed $i \
      --seed 74 \
      --load_in_4bit 1 \
      --bnb_4bit_use_double_quant 1 \
      --bnb_4bit_compute_dtype "float16" \
      --bnb_4bit_quant_type "nf4" \
      --r 16 \
      --lora_alpha 32 \
      --lora_dropout 0.05 \
      --epochs 10 \
      --batch_size 4 \
      --lr 5e-5 \
      --weight_decay 0.01 \
      --warmup_steps 20 \
      --saving_checkpoint 1 \
      --interp_method "linear" \
      --min_g 1e-10 < /dev/null > training_log_8b_fixed.out 2>&1 &

    echo "✅ Finished Seed $i"
    echo ""
done