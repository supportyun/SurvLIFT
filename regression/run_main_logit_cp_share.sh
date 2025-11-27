#!/bin/bash
# run_main.sh

python LIFT/survlift/regression/hazard_prediction/main_logit_cp_share.py \
  --model_name "meta-llama/Llama-3.2-1B" \
  --device "cuda:0" \
  --load_in_4bit True \
  --bnb_4bit_use_double_quant True \
  --bnb_4bit_compute_dtype "float16" \
  --bnb_4bit_quant_type "nf4" \
  --r 8 \
  --lora_alpha 32 \
  --lora_dropout 0.1 \
  --epochs 15 \
  --batch_size 4 \
  --lr 1e-5 \
  --weight_decay 0.01 \
  --warmup_steps 6 \
  --saving_checkpoint True \
  --seed 74 \
  --interp_method "linear" \
  --min_g 1e-10


# chmod +x LIFT/survlift/regression/hazard_prediction/run_main_logit_cp_share.sh
# LIFT/survlift/regression/hazard_prediction/run_main_logit_cp_share.sh