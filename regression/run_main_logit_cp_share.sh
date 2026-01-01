#!/bin/bash
set -e

# 이 스크립트가 있는 폴더로 이동 (경로 문제 원천 차단)
cd "$(dirname "$0")"

# Seed 1만 실행
for i in 1
do
    

    python main_logit_cp_share.py \
      --model_name "meta-llama/Llama-3.2-1B" \
      --data_seed $i \
      --seed 74 \
      --load_in_4bit True \
      --bnb_4bit_use_double_quant True \
      --bnb_4bit_compute_dtype "float16" \
      --bnb_4bit_quant_type "nf4" \
      --r 8 \
      --lora_alpha 16 \
      --lora_dropout 0.1 \
      --epochs 15 \
      --batch_size 4 \
      --lr 1e-5 \
      --weight_decay 0.01 \
      --warmup_steps 6 \
      --saving_checkpoint True \
      --interp_method "linear" \
      --min_g 1e-10

    echo "✅ Finished Seed $i"
    echo ""
done