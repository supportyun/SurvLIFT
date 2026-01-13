#!/bin/bash

# 마감 시간을 고려하여 Seed 1번 하나에 집중합니다.
for i in 1
do
    echo "=================================================="
    echo "🚀 Emergency Run for Assignment: Seed $i"
    echo "=================================================="

    python regression/main_logit_cp_share.py       --model_name "meta-llama/Llama-3.2-1B"       --data_seed $i       --seed 74       --load_in_4bit True       --bnb_4bit_use_double_quant True       --bnb_4bit_compute_dtype "float16"       --bnb_4bit_quant_type "nf4"       --r 8       --lora_alpha 32       --lora_dropout 0.1       --epochs 20       --batch_size 4       --lr 1e-4       --weight_decay 0.01       --warmup_steps 6       --saving_checkpoint True       --interp_method "linear"       --min_g 1e-10

    echo "✅ Finished Seed $i"
done
