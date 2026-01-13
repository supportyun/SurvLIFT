## KaplanMeierFitter: 나중에 잔차의 분포를 비모수적으로 추정하기 위해 사용.
import pandas as pd
import numpy as np
from lifelines import WeibullAFTFitter, KaplanMeierFitter
import os

def process_seed(seed):
    """
    특정 Seed의 데이터 파일을 읽어서 AFT 기반 타겟 데이터를 생성하고 저장하는 함수
    """
    # 1. 파일 경로 설정 (상대 경로 사용)
    input_filename = f"synthetic_pbc_data_N300_given_2_covariate_seed{seed}.csv"
    output_filename = f"target_data_for_aft_seed{seed}.csv"
    
    # [수정] 스크립트 파일의 위치를 기준으로 경로를 계산합니다.
    # 1. 현재 파일(make_target.py)이 있는 폴더 (SurvLIFT/regression)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # 2. 상위 폴더 (SurvLIFT)
    project_root = os.path.dirname(current_script_dir)
    
    # 3. 데이터 폴더와 결합 (SurvLIFT/data)
    input_path = os.path.join(project_root, "data", input_filename)
    output_path = os.path.join(project_root, "data", output_filename)
    
    # 파일 존재 여부 확인
    if not os.path.exists(input_path):
        print(f"[Skip] Seed {seed}: 파일을 찾을 수 없습니다. ({input_path})")
        return

    print(f"[Start] Seed {seed} 처리 중...")
    df = pd.read_csv(input_path)

    # 2. 로그 시간 변환 (Y = log(T))
    # log(0) 방지를 위해 아주 작은 값(1e-5) 추가
    df['log_time'] = np.log(df['time'] + 1e-5)

    # 3. 초기 Beta 추정 (Weibull AFT 모델 사용)
    aft = WeibullAFTFitter()
    # 실제 데이터에 있는 공변량들 (trt, age 등)
    covariates = ['age', 'trt', 'time', 'status'] 
    
    try:
        aft.fit(df[covariates], duration_col='time', event_col='status')
    except Exception as e:
        print(f"  - AFT 모델 적합 실패 (Seed {seed}): {e}")
        return

    # 예측된 로그 시간 (X * beta)
    predicted_log_time = np.log(aft.predict_expectation(df))

    # 4. 잔차(Residual) 계산: e_i = Y_i - X*beta
    residuals = df['log_time'] - predicted_log_time

    # 5. 잔차에 대한 Kaplan-Meier 적합 (오차 분포 추정)
    kmf = KaplanMeierFitter()
    kmf.fit(residuals, event_observed=df['status'])

    new_targets_log = []

    # 6. Buckley-James Imputation (결측치 대체)
    for idx, row in df.iterrows():
        if row['status'] == 1:
            # 관측된 데이터(uncensored): 실제 시간 그대로 사용
            target = row['log_time']
        else:
            # 중도절단 데이터(censored): 조건부 기대값 계산
            e_i = residuals[idx]
            
            # e_i 시점에서의 생존 확률 S(e_i)
            s_at_resid = kmf.predict(e_i)
            
            if s_at_resid > 0:
                # e_i보다 큰 구간(더 살았을 구간)의 생존 곡선 가져오기
                mask = kmf.survival_function_.index > e_i
                surv_curve_after = kmf.survival_function_[mask]
                
                if not surv_curve_after.empty:
                    # 잔여 수명 기대값 = (곡선 아래 면적) / 현재 생존확률
                    area = np.trapz(surv_curve_after['KM_estimate'], surv_curve_after.index)
                    expected_extra_resid = area / s_at_resid
                    
                    # 최종 타겟 = 관측된 시간 + 추가 예상 시간
                    target = row['log_time'] + expected_extra_resid
                else:
                    target = row['log_time'] # 정보 부족 시 관측 시간 사용
            else:
                target = row['log_time']
        
        new_targets_log.append(target)

    # 7. 결과 저장
    # 모델 학습용 (로그 스케일 & 원래 시간 스케일)
    df['target_log_time'] = new_targets_log
    df['target_time'] = np.round(np.exp(new_targets_log), 2) ## 소수 둘째자리까지 표현
    df['age'] = np.round(df['age'], 2)## 소수 둘째자리까지 표현
    # 필요한 컬럼만 저장 (원본 컬럼 + 타겟)
    cols_to_save = ['id', 'trt', 'age', 'time', 'status', 'target_time', 'target_log_time']
    df[cols_to_save].to_csv(output_path, index=False)
    
    print(f"[Done] Seed {seed} 완료! 저장 경로: {output_path}")


def main():
    print("=== 전체 Seed 데이터에 대한 Target 생성 시작 ===")
    # Seed 1부터 10까지 반복
    for seed in range(1, 11):
        process_seed(seed)
    print("=== 모든 작업 완료 ===")

if __name__ == "__main__":
    main()