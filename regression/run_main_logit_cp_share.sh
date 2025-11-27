#!/bin/bash
# run_main.sh

  # python main_logit_cp_share.py \
  # --model_name "meta-llama/Llama-3.2-1B" \
  # --device "cuda:0" \
  # --load_in_4bit True \
  # --bnb_4bit_use_double_quant True \
  # --bnb_4bit_compute_dtype "float16" \
  # --bnb_4bit_quant_type "nf4" \
  # --r 8 \
  # --lora_alpha 32 \
  # --lora_dropout 0.1 \
  # --epochs 15 \
  # --batch_size 4 \
  # --lr 1e-5 \
  # --weight_decay 0.01 \
  # --warmup_steps 6 \
  # --saving_checkpoint True \
  # --seed 74 \
  # --interp_method "linear" \
  # --min_g 1e-10


# chmod +x LIFT/survlift/regression/hazard_prediction/run_main_logit_cp_share.sh
# LIFT/survlift/regression/hazard_prediction/run_main_logit_cp_share.sh

#!/bin/bash

# [수정 전] for i in {1..10}
# [수정 후] seq 명령어로 1부터 10까지 숫자를 생성합니다.
for i in $(seq 1 10)
do
    echo "=================================================="
    echo "Running Training for Data Seed: $i"
    echo "=================================================="

    python main_logit_cp_share.py \
      --model_name "meta-llama/Llama-3.2-1B" \
      --data_seed $i \
      --seed 74 \
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
      --interp_method "linear" \
      --min_g 1e-10

    echo "Finished Data Seed $i"
    echo ""
done