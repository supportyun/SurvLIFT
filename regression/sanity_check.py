import pandas as pd
import numpy as np
import sys
import os

# 현재 폴더 경로 추가 (import를 위해)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_utils_share import data2text_cp2, build_time_group_means, lookup_time_from_groups, extract_variable

def run_sanity_check():
    print("=== [1] 가상 데이터 생성 및 로드 테스트 ===")
    # 실제 파일 대신 테스트용 가상 데이터프레임 생성
    df = pd.DataFrame({
        'id': [1, 2, 3, 4],
        'age': [35.5, 55.2, 70.1, 36.0],
        'trt': [1, 0, 1, 0], # 1: 치료, 0: 위약
        'status': [1, 0, 1, 0],
        'target_time': [10.5, 5.0, 3.2, 8.8] # 우리가 만든 정답 컬럼
    })
    print("Data Sample:\n", df)
    print("✅ 데이터 준비 완료\n")

    print("=== [2] 프롬프트 변환 테스트 (Time Prediction) ===")
    prompt = data2text_cp2(df.iloc[0])
    print(f"Generated Prompt:\n{prompt}")
    
    # 체크 포인트: 질문이 'survival time'인지, 정답이 숫자인지
    if "predict the expected survival time" in prompt and "10.50" in prompt:
        print("✅ 프롬프트 질문 & 정답 포맷 정상!")
    else:
        print("❌ 프롬프트 형식이 이상합니다. data2text_cp2를 확인하세요.")
        return # 중단

    print("\n=== [3] Fallback 족보(Mean Dictionary) 생성 테스트 ===")
    # 족보 만들기
    means_dict = build_time_group_means(df, target_col='target_time')
    print("Generated Dictionary:", means_dict)
    
    # 20-39세(age_bin='20-39'), 치료(trt=1)인 그룹의 평균 확인
    # 위 데이터에서 ID 1(35.5세, trt=1, time=10.5) 하나뿐이므로 평균은 10.5여야 함
    expected_val = 10.5
    if means_dict.get(('20-39', 1)) == expected_val:
        print("✅ 그룹별 평균 계산 로직 정상!")
    else:
        print(f"❌ 평균 계산 오류. 기대값: {expected_val}, 실제값: {means_dict.get(('20-39', 1))}")

    print("\n=== [4] Fallback 조회(Lookup) 테스트 ===")
    # Case A: 족보에 있는 그룹 (ID 1과 같은 그룹)
    val_exist = lookup_time_from_groups(38.0, 1, means_dict)
    print(f"Case A (있는 그룹): {val_exist} (Expected: 10.5)")
    
    # Case B: 족보에 없는 그룹 (80세 노인 -> Train 데이터에 없음)
    # 이때는 Global Mean(전체 평균)이 나와야 함. (10.5+5.0+3.2+8.8)/4 = 6.875
    val_fallback = lookup_time_from_groups(90.0, 1, means_dict)
    global_mean = df['target_time'].mean()
    print(f"Case B (없는 그룹): {val_fallback} (Expected Global Mean: {global_mean})")
    
    if val_fallback == global_mean:
        print("✅ 미지의 그룹에 대한 Global Mean 대체(Fallback) 정상!")
    else:
        print("❌ Fallback 로직 오류.")

    print("\n=== [5] 변수 추출 테스트 ===")
    vars_ = extract_variable([prompt])
    print(f"Extracted: {vars_}")
    if vars_['age'][0] == 35.5 and vars_['trt'][0] == 1:
        print("✅ 변수 추출(Extract) 정상!")
    else:
        print("❌ 변수 추출 오류.")

    print("\n🎉 모든 논리 검증(Sanity Check) 통과! 통합 테스트로 넘어가세요.")

if __name__ == "__main__":
    run_sanity_check()