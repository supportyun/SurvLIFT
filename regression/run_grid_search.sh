#!/bin/bash
set -e  # Stop script immediately on error
export CUDA_VISIBLE_DEVICES=1

# Move to the script's directory
cd "$(dirname "$0")"

# ==========================================
# 🔑 Token Settings
# ==========================================
export HF_TOKEN=

# ==========================================
# 🎛️ Experiment Settings (Grid Search)
# ==========================================
SEEDS=(1 2)                 
RANKS=(16 32)               
LRS=("1e-4" "5e-5")         
# ==========================================

echo "🚀 Starting Grid Search: Total 8 Experiments"
echo "---------------------------------------------"

# Counter for progress tracking
count=0
total=8

for seed in "${SEEDS[@]}"; do
    for r in "${RANKS[@]}"; do
        for lr in "${LRS[@]}"; do
            
            # Increment counter
            count=$((count + 1))

            # --- [Auto-Parameter Logic] ---
            alpha=$((r * 2))
            
            if [ "$r" -eq 32 ]; then
                dropout=0.1
            else
                dropout=0.05
            fi
            # ----------------------------------

            log_file="logs/training_s${seed}_r${r}_lr${lr}.out"
            mkdir -p logs

            echo ""
            echo "=========================================================="
            echo "▶️  [Progress: $count / $total] Starting Experiment..."
            echo "   • Settings: Seed=$seed | Rank=$r | Alpha=$alpha | Drop=$dropout | LR=$lr"
            echo "   • Log File: $log_file"
            echo "=========================================================="

            # ✅ KEY CHANGE: Using 'tee' to show output AND save to file
            # PYTHONUNBUFFERED=1 prevents print delay
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

            echo ""
            echo "✅ [Experiment Done] ($count / $total) Finished!"
            echo "---------------------------------------------"
            
            sleep 5
        done
    done
done

echo "🎉 All Grid Search Experiments Finished!"