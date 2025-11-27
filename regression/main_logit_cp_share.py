# main.py
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import argparse
import torch
import pandas as pd
import numpy as np
from llama_finetuner_logit_cp import LlamaFinetuner
from sklearn.model_selection import train_test_split
from data_utils_share import *
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
from lifelines.utils import concordance_index
from transformers import set_seed


def prepare_data():
    df = pd.read_csv("/home/kanggi1/survlift/LIFT/survlift/data/synthetic_pbc_data_N300_given_2_covariate_seed1.csv")
    
    event_data = df[df['status'] == 1]
    censor_data = df[df['status'] == 0]

    event_train, event_temp = train_test_split(event_data, test_size=0.2, random_state=42)
    event_validate, event_test = train_test_split(event_temp, test_size=0.5, random_state=42)
    censor_train, censor_temp = train_test_split(censor_data, test_size=0.2, random_state=42)
    censor_validate, censor_test = train_test_split(censor_temp, test_size=0.5, random_state=42)

    train = pd.concat([event_train, censor_train]).sample(frac=1, random_state=42)
    validate = pd.concat([event_validate, censor_validate]).sample(frac=1, random_state=42)
    test = pd.concat([event_test, censor_test]).sample(frac=1, random_state=42)

    return train, validate, test



def main(args):
    print("Prepare data...")
    set_seed(args.seed)
    train, validate, test = prepare_data()
    
    unique_validate = validate[["id", "time", "status"]]
    unique_validate[['time']] = unique_validate[['time']].round(2)
    
    unique_test = test[["id","time","status"]]
    unique_test[['time']] = unique_test[['time']].round(2)
        
    train_df = convert_to_counting_process_uniq(train)
    train_df = calculate_hazard(train_df, rho=1.5)
    train_df[['age','time']] = train_df[['age','time']].round(2)
    train_df[['hazard']] = train_df[['hazard']].round(4)
        
    validate_df = convert_to_counting_process_uniq2(validate)
    validate_df = calculate_hazard(validate_df, rho=1.5)
    validate_df[['age','time']] = validate_df[['age','time']].round(2)
    validate_df[['hazard']] = validate_df[['hazard']].round(4)
        
    test_df = convert_to_counting_process_uniq2(test)
    test_df = calculate_hazard(test_df, rho=1.5)
    test_df[['age','time']] = test_df[['age','time']].round(2)
    test_df[['hazard']] = test_df[['hazard']].round(4)

    init= ''
    end = ''

    train_prompts = df2prompts(train_df, data2text_cp2, init, end)
    val_prompts = df2prompts(validate_df, data2text_cp2, init, end)
    test_prompts = df2prompts(test_df, data2text_cp2, init, end)


    print("Save data...")
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results/250923/data1/seed_{args.seed}")
    os.makedirs(base_dir, exist_ok=True)
    output_dir = os.path.join(base_dir, f"epochs_{args.epochs}_lr_{args.lr}")
    os.makedirs(output_dir, exist_ok=True)

    train_js_path = os.path.join(output_dir, "synthetic_prompts_train.jsonl")
    val_js_path   = os.path.join(output_dir, "synthetic_prompts_val.jsonl")
    test_js_path  = os.path.join(output_dir, "synthetic_prompts_test.jsonl")

    train_js = write_jsonl('\n'.join(train_prompts), train_js_path)
    val_js   = write_jsonl('\n'.join(val_prompts), val_js_path)
    test_js  = write_jsonl('\n'.join(test_prompts), test_js_path)
    
    fallback_means = build_group_means(train_df, t_col="time", hazard_col="hazard",
                                   age_col="age", trt_col="trt")
    
    
    print('Start train...')
    finetuner = LlamaFinetuner(
        model_name=args.model_name,
        device=args.device,
        output_dir=output_dir,
        load_in_4bit=args.load_in_4bit,
        # load_in_8bit=args.load_in_8bit,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        seed = args.seed
    )

    finetuner.train(train_js, val_js,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    warmup_steps=args.warmup_steps,
                    saving_checkpoint=args.saving_checkpoint,
                    unique_validate_df = unique_validate, 
                    interp_method = args.interp_method,
                    min_g = args.min_g,
                    fallback_means = fallback_means
                    )
    
    
    
    print("Start evaluate...")
    finetuner.load_model()
    
    test_prompts = extract_prompts(test_js, '')
    test_completions = extract_completion(test_js)

    ids = extract_id(test_prompts)
    times = extract_time2(test_prompts)
    
    ans, invalid_ratio, final_invalid_ratio = finetuner.generate(text_lst=test_prompts, max_token=10, batch_size=args.batch_size, valid_mean = 0.0)

    print("test answer: ", ans)

    test_truth = [float(s.split('@@@', 1)[0].strip()) for s in test_completions]
    
    df = pd.DataFrame({
        'ids': ids,
        'times': times,
        'hazard_true': test_truth,
        'hazard_pred': ans
    })
    
    df_with_survival = cumulative_hazard_trap_scipy(df, id_col="ids", time_col="times", hazard_col="hazard_pred")
    
    df_with_cumhaz = df_with_survival.merge(
        test_df[['id', 'time', 'hazard', 'cum_haz']],
        how = 'left',
        left_on=["ids", "times", "hazard_true"], 
        right_on=["id", "time", "hazard"]      
    ).drop(columns=["id", "time", "hazard"])
    
    df_unique = df_with_survival.merge(
        unique_test[['id','time','status']],
        left_on=['ids','times'],  
        right_on=['id','time'],   
        how='inner'
    ).drop(columns=['id','time']) \
    .rename(columns={'ids':'id', 'times':'time'})
    

    df_with_cumhaz.to_csv(os.path.join(output_dir, "cumhaz.csv"), index=True)
    df_with_survival.to_csv(os.path.join(output_dir, "survival.csv"), index=True)
    df_unique.to_csv(os.path.join(output_dir, "unique_df.csv"), index=True)
    
    id_list = sorted([i for i in df_with_survival['ids'].unique()])
    n_rows, n_cols = 8, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 25))
    axes = np.atleast_2d(axes)

    color_true = '#5FA8AF'
    color_pred = 'orange'

    total_slots = n_rows * n_cols

    legend_slot = 0 
    plot_slots = [i for i in range(total_slots) if i != legend_slot]

    for this_id, slot in zip(id_list, plot_slots):
        r, c = divmod(slot, n_cols)
        ax = axes[r, c]

        llm_plot_df = df_with_survival[df_with_survival["ids"] == this_id].sort_values(by="times")
        mae = calculate_mae(llm_plot_df["hazard_true"], llm_plot_df["hazard_pred"]).round(4)
        ax.plot(llm_plot_df["times"], llm_plot_df["hazard_true"],
                label="True", marker="o", color=color_true, zorder=2)
        ax.plot(llm_plot_df["times"], llm_plot_df["hazard_pred"],
                label="Predicted", marker="o", color=color_pred, zorder=2)

        ax.set_title(f"ID {int(this_id)} (MAE = {mae})")
        ax.set_xlabel("Time in year")
        ax.set_ylabel("Hazard")
        ax.grid(True)
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)

        this_df = df_unique[df_unique['id'] == this_id]
        if this_df['status'].iloc[0] == 1:
            last_time = this_df['time'].iloc[0]
            ax.axvline(x=last_time, color='#D63F29', linestyle='-', linewidth=2)

    leg_ax = axes.flat[legend_slot]
    leg_ax.axis('off')
    handles = [
        Line2D([], [], linestyle='-', marker='o', color=color_true, label='True', lw=2, ms=8),
        Line2D([], [], linestyle='-', marker='o', color=color_pred, label='Predicted', lw=2, ms=8),
        Line2D([], [], linestyle='-', color='#D63F29', label='Event time', lw=2, ms=8)
    ]
    leg_ax.legend(handles=handles, loc='center', frameon=True, fontsize=20)

    unused_slots = plot_slots[len(id_list):]
    for k in unused_slots:
        fig.delaxes(axes.flat[k])

    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(output_dir, "hazard_plot.png"), dpi=500)

    S_at_T = df_unique['S'].tolist()
    risk_scores = [-s for s in S_at_T]
    c_index = concordance_index(df_unique['time'].tolist(), risk_scores, df_unique['status'].tolist())
    
    event_df = unique_test.rename(columns={'time': 'Y', 'status': 'delta'}).copy()
    hz_use = df_with_survival[['ids', 'times', 'S']].copy()
    hz_use = hz_use.rename(columns={'ids': 'id', 'times': 'time', 'S': 'survival_prob'})
    BS_df = brier_ipcw(hz_use, event_df[['id','Y','delta']], method=args.interp_method, min_g=args.min_g, model = "SurvLIFT")
    ibs = ibs_from_bs(BS_df)
    BS_df.to_csv(os.path.join(output_dir, "brier_ipcw_over_time.csv"), index=False)
    with open(os.path.join(output_dir, "ibs.txt"), "w") as f:
        f.write(str(ibs) + "\n")
    
    print("C-index: ", c_index)
    print("IBS:", ibs)
    print("Ratio of invalid answer: ", invalid_ratio)
    print("Final invalid ratio: ", final_invalid_ratio)

    def save_test_metrics(output_dir, c_index, ibs, none_ratio, final_invalid_ratio):
        metrics = {
            "test_c_index": c_index,
            "test_ibs": ibs, 
            "invalid_answer_ratio":none_ratio,
            "fianl_invalid_anser_ratio": final_invalid_ratio
        }
        path = os.path.join(output_dir, "test_metrics.json")
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
    
    save_test_metrics(output_dir, c_index, ibs, invalid_ratio, final_invalid_ratio)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--system_message", type=str, default="You are a helpful and knowledgeable assistant. Answer as concisely as possible.")

    parser.add_argument("--load_in_4bit", type=bool, default=True)
    # parser.add_argument("--load_in_8bit", type=bool, default=False)
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
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--interp_method", type=str, default="locf")
    parser.add_argument("--min_g", type=float, default=1e-10)
    args = parser.parse_args()
    
    args.device = torch.device(args.device)
    
    main(args)
