# data_utils.py
import os
import json
import pandas as pd
import numpy as np
from functools import partial
import re
from sklearn.metrics import mean_absolute_error
from scipy.integrate import cumulative_trapezoid
from typing import Optional
from lifelines import KaplanMeierFitter

def convert_to_counting_process_uniq(df):
    uniq = np.unique(df['time']).tolist()
    new_rows = []
    
    for idx, row in df.iterrows():
        id_val = int(row['id'])
        time_val = row['time']
        status_val = row['status']
        
        covariates = row.drop(['id', 'time', 'status']).to_dict()
        for t in uniq:
            if t > time_val:
                break 
            else:
                new_row = {
                    'id': int(id_val),
                    'time': t,
                    'status': status_val if t == time_val else 0 
                }
            new_row.update(covariates)
            new_rows.append(new_row)
    
    out = pd.DataFrame(new_rows)
    out['id'] = out['id'].astype('int64')

    return out


def convert_to_counting_process_uniq2(df):
    uniq = np.unique(df['time']).tolist()
    new_rows = []
    
    for idx, row in df.iterrows(): 
        id_val = int(row['id'])
        time_val = row['time']
        status_val = row['status']

        covariates = row.drop(['id', 'time', 'status']).to_dict()
        for t in uniq:
            if t > time_val:
                if status_val == 1:
                    new_row = {
                        'id': id_val,
                        'time': t,
                        'status': '!'
                    }
                else:
                    new_row = {
                        'id': id_val,
                        'time': t,
                        'status': '?'
                    }
            else:
                new_row = {
                    'id': id_val,
                    'time': t,
                    'status': status_val if t == time_val else 0
                }
            
            new_row.update(covariates)
            new_rows.append(new_row)
    
    return pd.DataFrame(new_rows)

def calculate_hazard(df, rho):
    df["hazard"] = (rho / np.power(df["new_lambda"], rho)) * \
               np.power(df["time"], rho - 1)
    df["cum_haz"] = (1 / np.power(df["new_lambda"], rho)) * \
               np.power(df["time"], rho)

    return df

def df2prompts(df, data2text_func, init='', end=''):
    jsonl = df.apply(func=partial(data2text_func, init=init, end=end), axis=1).tolist()
    return jsonl

def data2text_cp2(row, label=True, init='', end=''):
    prompt = init
    status = row['status']
    phrase = (
        'was still alive at' if status == 0 else
        'died at'            if status == 1 else
        'was already off study by' if status == '?' else
        'had already died by'
    )
    prompt += (
        f"The patient with id {int(row['id']):d} was enrolled in the primary biliary cholangitis (PBC) study. At the enrollment period, "
        f"the patient was {row['age']} years old, and {'was treated with D-penicillamine' if row['trt'] == 1 else 'received no active treatment (placebo)'}. "
        f"Note that this patient {phrase} year {row['time']}. "
        f"Based on these observed values, what is the instantaneous hazard for this patient at time {row['time']} years? "
        f"By 'instantaneous hazard' we mean the event rate at time t, conditional on having survived up to t. "
        f"Output only one non-negative real number with exactly 4 decimals. No extra text."
    )
    prompt += end

    if not label:
        final_prompt = f"{prompt}###"
    else:
        completion = row['hazard']
        final_prompt = "{\"prompt\":\"%s###\", \"completion\":\"%s@@@\"}" % (prompt, completion)
    return final_prompt

def write_jsonl(jsonl, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        f.write(jsonl)
    return filename

def bin_age(age: float) -> str:
    if 20 <= age < 40: return "20-39"
    if 40 <= age < 60: return "40-59"
    if 60 <= age < 80: return "60-79"
    return "out"

def build_group_means(train_df: pd.DataFrame,
                      t_col="time", hazard_col="hazard",
                      age_col="age", trt_col="trt"):
    
    df = train_df.copy()
    df["age_bin"] = df[age_col].apply(bin_age)

    by_full = (df.groupby(["age_bin", trt_col, t_col])[hazard_col]
                 .mean().reset_index())
    by_full = { (r["age_bin"], int(r[trt_col]), r[t_col]): float(r[hazard_col])
                for _, r in by_full.iterrows() }

    return {
        "by_full": by_full
    }


def extract_prompts(jsonl_file, in_context_prefix=''):
    test_prompts = []
    with open(jsonl_file) as fp:
        for line in fp:
            json_obj = json.loads(line)
            test_prompts.append(in_context_prefix + json_obj['prompt'])
    return test_prompts


def extract_completion(jsonl_file):
    completions = []
    with open(jsonl_file) as fp:
        for line in fp:
            json_obj = json.loads(line)
            completions.append(json_obj['completion'])
    return completions


def extract_id(prompts):
    ids = []
    for prompt in prompts:
        match = re.search(r"The patient with id (\d+)", prompt)
        if match:
            ids.append(int(match.group(1)))
    return ids


def extract_time2(prompts):
    times = []
    for prompt in prompts:
        match = re.search(r"at time ([\d\.]+)", prompt)
        if match:
            times.append(float(match.group(1).rstrip('.')))
    return times


def cumulative_hazard_trap_scipy(df, id_col="id", time_col="time", hazard_col="lambda"):
    df = df.sort_values([id_col, time_col]).copy()
    
    def _one(g):
        t = g[time_col].to_numpy()
        lam = np.clip(g[hazard_col].to_numpy(), 0, None)
        H = np.r_[0.0, cumulative_trapezoid(lam, t)]  # H(t_k)
        g["H"] = H
        g["dH"] = np.r_[H[0], np.diff(H)]
        g["S"] = np.exp(-H)
        return g
    
    return df.groupby(id_col, group_keys=False).apply(_one)


def calculate_mae(list1, list2):

    if len(list1) != len(list2):
        raise ValueError("The two lists must be of the same length.")
    
    return mean_absolute_error(list1, list2)


def fit_km_censor(event_df):
    kmf_c = KaplanMeierFitter()
    kmf_c.fit(durations=event_df['Y'].values,
              event_observed=(1 - event_df['delta'].values))
    
    timeline = kmf_c.survival_function_.index.values
    surv_vals = kmf_c.survival_function_["KM_estimate"].values

    def step_eval(ts):
        ts = np.asarray(ts, dtype=float)
        idx = np.searchsorted(timeline, ts, side="right") - 1
        out = np.ones_like(ts, dtype=float)
        mask = idx >= 0
        out[mask] = surv_vals[idx[mask]]
        return out

    return kmf_c, step_eval


def prepare_pred_matrix(hazard_df, event_df, method="locf", model="SurvLIFT"):
    tau = float(np.median(event_df['Y']))
    print("median time: ", tau)
    times_eval = np.arange(0, tau + 1, 0.05)

    if model == "SurvLIFT":
        hz = hazard_df.sort_values(['id','time'])
        S = hz.pivot_table(index='id', columns='time', values='survival_prob', aggfunc='last')
    else:
        S = hazard_df

    S[0.0] = 1.0     
    S = S.sort_index(axis=1)  
    S = S.reindex(columns=times_eval, method='ffill')

    if method == "locf":
        S = S.ffill(axis=1) 

    elif method == "linear":
        S = S.apply(lambda row: row.interpolate(method="values", limit_direction="forward"), axis=1)
        
    else:
        raise ValueError("method must be 'locf' or 'linear'")
    
    ev = event_df.set_index('id')
    S, ev = S.align(ev[['Y']], join='inner', axis=0)
    
    assert set(S.index) <= set(event_df['id']), "Hazard_df has an ID that is not in the test set."
    
    return times_eval, S


def brier_ipcw(hazard_df, event_df, method="locf", min_g=1e-10, model = "SurvLIFT"):
    _, Ghat = fit_km_censor(event_df)
    times_eval, S = prepare_pred_matrix(hazard_df, event_df, method=method, model=model)

    ev = event_df.set_index('id').loc[S.index].copy()
    Y = ev['Y'].values
    delta = ev['delta'].values

    out_time, out_bs, out_n = [], [], []
    for j, t in enumerate(times_eval):
        S_t = S.iloc[:, j].values
        avail = ~np.isnan(S_t) 
        if not np.any(avail):
            continue

        S_av = S_t[avail]
        Y_av = Y[avail]
        d_av = delta[avail]

        GY = np.maximum(Ghat(Y_av), min_g)  
        Gt = max(Ghat([t])[0], min_g)        

        mask1 = (Y_av <= t) & (d_av == 1)
        term1 = np.sum(((0.0 - S_av[mask1]) ** 2) / GY[mask1]) if np.any(mask1) else 0.0

        mask2 = (Y_av > t)
        term2 = np.sum(((1.0 - S_av[mask2]) ** 2) / Gt) if np.any(mask2) else 0.0

        denom = np.sum(avail)
        
        out_time.append(t)
        out_bs.append((term1 + term2) / denom)
        out_n.append(denom)

    return pd.DataFrame({'time': out_time, 'BS': out_bs, 'n_eff': out_n})


def ibs_from_bs(bs_df):
    if bs_df is None or len(bs_df) < 2:
        return np.nan
    t = bs_df['time'].values
    b = bs_df['BS'].values
    
    if t[-1] == t[0]:
        return np.nan
    
    return np.trapz(b, t) / (t[-1] - t[0])

def parse_number_4dec(text: str,
                      stop_str: str = "@@@",
                      which: str = "last",       
                      clip_min: Optional[float] = None,
                      clip_max: Optional[float] = None):
    
    if stop_str and stop_str in text:
        text = text.split(stop_str, 1)[0]
    
    s = text.strip()
    
    try:
        val = float(s)
    except ValueError:
        return None
    if clip_min is not None and val < clip_min:
        val = clip_min
    if clip_max is not None and val > clip_max:
        val = clip_max
    
    return val

def extract_variable(prompts):
    AGE_RE   = re.compile(r'\b(\d+(?:\.\d+)?)\s*years?\s*old\b', re.IGNORECASE)
    TIME_RE  = re.compile(r'\bat\s*time\s*(\d+(?:\.\d+)?)\s*years?\b', re.IGNORECASE)
    
    TRT1_RE  = re.compile(r'\bD[\-\u2010-\u2015]?penicillamine\b', re.IGNORECASE)
    TRT2_RE  = re.compile(r'\bplacebo\b', re.IGNORECASE)
    
    ages, trts, times = [], [], []
    for p in prompts:
        m_age = AGE_RE.search(p)
        ages.append(float(m_age.group(1)) if m_age else None)

        if TRT1_RE.search(p):
            trt_val = 1
        elif TRT2_RE.search(p):
            trt_val = 2
        else:
            trt_val = None  
        trts.append(trt_val)

        m_time = TIME_RE.search(p)
        times.append(float(m_time.group(1)) if m_time else None)

    return {"age": ages, "trt": trts, "t": times}

def fallback_lookup_from_means(age, trt, t_key, means, clip_min=None, clip_max=None):
    age_bin = bin_age(age)
    trt = int(trt)

    val = None
    by_full = means.get("by_full", {})

    if t_key is not None:
        val = by_full.get((age_bin, trt, t_key))

    if val is None and t_key is not None:
        candidates = [k[2] for k in by_full.keys()
                      if k[0] == age_bin and k[1] == trt and k[2] < t_key]
        if candidates:
            best_t = max(candidates)
            val = by_full.get((age_bin, trt, best_t))

    if clip_min is not None and val < clip_min:
        val = clip_min
    if clip_max is not None and val > clip_max:
        val = clip_max

    return float(val)