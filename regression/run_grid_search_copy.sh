#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=1

cd "$(dirname "$0")"

# ==========================================
# 🔐 Token (터미널에서 export 후 실행 권장)
# ==========================================


# ==========================================
# ⚔️ The Final Battle: Rank 16 vs Rank 32
# ==========================================
SEEDS=(1 2 3 4 5)       # 통계적 유의성을 위해 5개
RANKS=(16 32)           # 2순위와 1순위 모델 둘 다 비교
LRS=("5e-5")            # LR은 이게 베스트였음 (고정)
# ==========================================

echo "🚀 Starting Final Validation: Total $(( ${#SEEDS[@]} * ${#RANKS[@]} )) Experiments"

count=0
total=$(( ${#SEEDS[@]} * ${#RANKS[@]} ))

for seed in "${SEEDS[@]}"; do
    for r in "${RANKS[@]}"; do
        for lr in "${LRS[@]}"; do
            
            count=$((count + 1))

            # --- [Auto-Parameter Logic] ---
            # Rank에 따라 Alpha와 Dropout을 자동으로 조절
            alpha=$((r * 2))
            
            if [ "$r" -eq 32 ]; then
                dropout=0.1   # Rank 32일 때 최적
            else
                dropout=0.05  # Rank 16일 때 최적
            fi
            # ----------------------------------

            log_file="logs/final_s${seed}_r${r}_lr${lr}.out"
            mkdir -p logs

            echo ""
            echo "=========================================================="
            echo "▶️  [Progress: $count / $total] Running Experiment..."
            echo "   • Seed: $seed | Rank: $r | Alpha: $alpha | LR: $lr"
            echo "=========================================================="

            export PYTHONUNBUFFERED=1
            
            python main_logit_cp_share.py \
              --model_name "meta-llama/Llama-3.1-8B" \
              --data_seed $seed \
              --seed 74 \
              --load_in_4bit 1 \
              --bnb_4bit_use_double_quant 1 \
              --bnb_4bit_compute_dtype "float16" \
              --bnb_4bit_quant_type "nf4" \
              --r $r \
              --lora_alpha $alpha \
              --lora_dropout $dropout \
              --epochs 15 \
              --batch_size 4 \
              --lr $lr \
              --weight_decay 0.01 \
              --warmup_steps 20 \
              --saving_checkpoint 1 \
              --interp_method "linear" \
              --min_g 1e-10 \
              2>&1 | tee "$log_file"
            
            sleep 5
        done
    done
done

echo "🎉 All Experiments Finished!"