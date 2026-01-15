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

#(data2text_func함수를 행별로 적용해서 jsonl 형식으로 뱉어준다)
def df2prompts(df, data2text_func, init='', end=''):
    jsonl = df.apply(func=partial(data2text_func, init=init, end=end), axis=1).tolist()
    return jsonl
#partial 함수를 써서 미리 init, end를 지정한채 실행, axis=1로 한 행씩 appy함수를 적용한다
def data2text_cp2(row, label=True, init='', end=''): 
    ###label=True라는 것은 학습 모드라는 것, 나중에 main_logit_cp_share에서 test, train data를 만들 때 사용됨.
    prompt = init
    # status = row['status']
    # phrase = (
    #     'was still alive at' if status == 0 else
    #     'died at'            if status == 1 else
    #     'was already off study by' if status == '?' else
    #     'had already died by'
    # )
    # 이 부분은 hazard 예측에서는 환자가 t 시점에 어떤 상태였는지에 대한 정보가 필요하지만 지금은 기대수명을 묻기 때문에 정보로 활용되지 않음.
    prompt += (
        f"The patient with id {int(row['id']):d} was enrolled in the PBC study. "
        f"At enrollment, the patient was {row['age']:.2f} years old, and "
        f"{'was treated with D-penicillamine' if row['trt'] == 1 else 'received placebo'}. "
        # [중요] 질문을 '로그 시간'이 아닌 '실제 생존 시간(T)' 예측으로 변경
        f"Based on these features, predict the expected survival time (T) for this patient. "
        f"Output only one non-negative real number. No extra text."
    )

    prompt += end

    if not label: ### 추론모드라는 의미이다.
        final_prompt = f"{prompt}###"
    else:
        # [중요] 정답은 make_target에서 exp 취한 target_time (소수점 2자리)
        completion = f"{row['target_time']:.2f}" 
        final_prompt = "{\"prompt\":\"%s###\", \"completion\":\"%s@@@\"}" % (prompt, completion)
    return final_prompt

# 만든 프롬프트 문자열을 실제 파일 jsonl로 저장하는 함수
def write_jsonl(jsonl, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True) #지정된 경로에 폴더가 없으면 알아서 만들고 이미 파일이 있으면 에러내지말고 그냥 넘어가기
    with open(filename, 'w') as f:
        f.write(jsonl)
    return filename
#연령별 bin 만들기
def bin_age(age: float) -> str:
    if 20 <= age < 40: return "20-39"
    if 40 <= age < 60: return "40-59"
    if 60 <= age < 80: return "60-79"
    return "out"

#LLM이 예측을 실패했을 때 대신 내놓는 값을 딕셔너리 형태로 반환함
def build_time_group_means(train_df, target_col="target_time", age_col="age", trt_col="trt"):
    df = train_df.copy()
    df["age_bin"] = df[age_col].apply(bin_age) # bin_age 함수는 기존 그대로 사용

    # [수정] 나이와 치료법별로 '생존 시간(Target)'의 평균을 구함 (Linear Scale)
    by_full = df.groupby(["age_bin", trt_col])[target_col].mean().reset_index()
    #(.reset_index는 dataframe으로 변환시켜줌)
    
    # 위에서 만든 by_full을 딕셔너리 형태로 변환시켜줌
    means_dict = {
        # [수정] 그룹 평균 반올림 (소수점 2자리)
        (r["age_bin"], int(r[trt_col])): round(float(r[target_col]), 2)
        for _, r in by_full.iterrows() #_자리에는 index가 들어가고 r은 모든 데이터 뭉치
    }
    
    # [수정] 전체 평균(Global Mean) 반올림 (소수점 2자리)
    means_dict["global_mean"] = round(df[target_col].mean(), 2)
    
    return means_dict

#jsonl: {"prompt": "The patient with id 1... predict the expected survival time (T)...###", "completion": "10.50@@@"}
#저장된 jsonl에서 프롬프트만 추출; test할 때 문제지만 입력으로 넣어주는 용도
def extract_prompts(jsonl_file, in_context_prefix=''):
    #in_context_prefix: few-shot prompting을 위해 앞에 몇개의 예시를 붙여줄 수 있음
    test_prompts = []
    with open(jsonl_file) as fp:
        for line in fp:
            json_obj = json.loads(line)
            test_prompts.append(in_context_prefix + json_obj['prompt'])
    return test_prompts

#저장된 jsonl에서 정답 추출; RMSE 또는 MAE를 계산할 때 비교 기준으로 사용함
def extract_completion(jsonl_file):
    completions = []
    with open(jsonl_file) as fp:
        for line in fp:
            json_obj = json.loads(line)
            completions.append(json_obj['completion'])
    return completions

#extract_prompts를 이용해서 추출한 prompt에서 id만 뽑아냄
def extract_id(prompts):
    ids = []
    for prompt in prompts:
        match = re.search(r"The patient with id (\d+)", prompt)
        if match:
            ids.append(int(match.group(1))) #match.group(1)은 id의 문자열 버전, 괄호에 해당하는 값 불러옴
    return ids

#extract_prompts를 이용해서 추출한 prompt에서 시간을 뽑아내는건데 이 경우에는 안씀
# def extract_time2(prompts):
#     times = []
#     for prompt in prompts:
#         match = re.search(r"at time ([\d\.]+)", prompt)
#         if match:
#             times.append(float(match.group(1).rstrip('.')))
#     return times


def calculate_mae(list1, list2):

    if len(list1) != len(list2):
        raise ValueError("The two lists must be of the same length.")
    
    return mean_absolute_error(list1, list2)

##### 환자가 중도절단될 확룔 G(t)를 구하는 함수이다.
# def fit_km_censor(event_df):
#     kmf_c = KaplanMeierFitter()
#     kmf_c.fit(durations=event_df['Y'].values,
#               event_observed=(1 - event_df['delta'].values))
    
#     timeline = kmf_c.survival_function_.index.values
#     surv_vals = kmf_c.survival_function_["KM_estimate"].values

#     def step_eval(ts):
#         ts = np.asarray(ts, dtype=float)
#         idx = np.searchsorted(timeline, ts, side="right") - 1
#         out = np.ones_like(ts, dtype=float)
#         mask = idx >= 0
#         out[mask] = surv_vals[idx[mask]]
#         return out

#     return kmf_c, step_eval

#모델이 예측한 결과인 문자를 실수 형태로 바꿔주는 함수

# [수정] Linear Scale을 고려하여 기본 Range 수정 및 파싱 후 Rounding 적용
def parse_number_4dec(text: str,
                      stop_str: str = "@@@",
                      which: str = "last",       
                      clip_min: Optional[float] = 0.0,    # [수정] 시간은 음수 불가
                      clip_max: Optional[float] = 120.0): # [수정] 사람 기대수명 상한(년 단위 가정)
    # 1. stop_str(@@@)이 있으면 그 뒤는 과감히 자릅니다.
    # 예: "21.5@@@Compute..." -> "21.5"
    if stop_str and stop_str in text:
        text = text.split(stop_str, 1)[0]
    
    # 2. [핵심 수정] 텍스트가 깨끗한 숫자가 아니어도(예: "Answer: 21.5") 찾아냅니다.
    # 정규표현식: 실수(float) 또는 정수 패턴 검색
    # r"[-+]?\d*\.\d+|\d+" : 음수 부호, 소수점 포함 숫자 탐색
    match = re.search(r"[-+]?\d*\.\d+|\d+", text)
    
    if match:
        try:
            val = float(match.group())
        except ValueError:
            return None
    else:
        # 숫자가 아예 없는 경우
        return None

    # 3. 범위 제한 (Clipping) - 기존 로직 유지
    if clip_min is not None and val < clip_min:
        val = clip_min
    if clip_max is not None and val > clip_max:
        val = clip_max
    
    # [수정] 파싱된 결과는 무조건 소수점 2자리로 반올림
    return round(val, 2)


##prompt를 읽어서 원래 환자의 나이, 치료법을 딕셔너리 형태로 추출하는 함수
def extract_variable(prompts):
    # 정규식은 기존과 동일
    AGE_RE = re.compile(r'\b(\d+(?:\.\d+)?)\s*years?\s*old\b', re.IGNORECASE)
    TRT1_RE = re.compile(r'\bD[\-\u2010-\u2015]?penicillamine\b', re.IGNORECASE)
    TRT2_RE = re.compile(r'\bplacebo\b', re.IGNORECASE)
    
    ages, trts = [], []
    for p in prompts:
        m_age = AGE_RE.search(p)
        ages.append(float(m_age.group(1)) if m_age else None)

        if TRT1_RE.search(p): trt_val = 1
        elif TRT2_RE.search(p): trt_val = 2
        else: trt_val = None
        trts.append(trt_val)
        
    return {"age": ages, "trt": trts}

#######이 부분 확인 받기##########
def lookup_time_from_groups(age, trt, means_dict):
    age_bin = bin_age(age)
    # trt 없으면 그냥 글로벌 평균 넣기
    if trt is None: return means_dict.get("global_mean")
    trt = int(trt)

    # 1. 그룹 평균 찾기
    val = means_dict.get((age_bin, trt))
    
    # 2. 없으면 전체 평균 사용
    if val is None:
        val = means_dict.get("global_mean")
        
    return float(val)
