# import pandas as pd
# import os

# def calculate_mcq_averages(fine_path, coarse_path, output_path):
#     # 1. 파일 존재 확인
#     if not os.path.exists(fine_path) or not os.path.exists(coarse_path):
#         print("❌ 파일을 찾을 수 없습니다.")
#         return

#     # 2. CSV 로드
#     df_fine = pd.read_csv(fine_path)
#     df_coarse = pd.read_csv(coarse_path)

#     # 3. 병합 (filename 기준)
#     # 두 파일의 열 이름이 같으므로 suffixes를 지정해 구분합니다.
#     merged = pd.merge(df_fine, df_coarse, on='filename', suffixes=('_fine', '_coarse'))

#     # 4. 평균 계산 (Vectorized Operation)
#     cols = ['mcq_type1_lighten_orig', 'mcq_type1_orig_orig', 'mcq_type1_orig_darken']
    
#     for col in cols:
#         # (Fine 컬럼 + Coarse 컬럼) / 2
#         merged[col] = (merged[f'{col}_fine'] + merged[f'{col}_coarse']) / 2

#     # 5. 결과 필터링 (filename과 계산된 평균 열만 추출)
#     final_df = merged[['filename'] + cols]

#     # 6. 저장
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     final_df.to_csv(output_path, index=False)

#     print(f"✅ 처리가 완료되었습니다.")
#     print(f"📍 저장 위치: {output_path}")

# if __name__ == "__main__":
#     base_path = "/data1/joo/pai_bench/result/prelim_01/metric/format/size"
#     fine_csv = os.path.join(base_path, "mcq_type1_fine.csv")
#     coarse_csv = os.path.join(base_path, "mcq_type1_coarse.csv")
#     output_csv = os.path.join(base_path, "mcq_type1.csv")

#     calculate_mcq_averages(fine_csv, coarse_csv, output_csv)
import pandas as pd

# 파일 경로 설정
file_path = "/data1/joo/pai_bench/result/prelim_01/metric/format/brightness/mcq_type1_baseline.csv"

try:
    # 1. CSV 파일 로드
    df = pd.read_csv(file_path)

    # 2. filename 열 제외 (수치 계산을 위해)
    # numeric_only=True를 설정하여 숫자 데이터만 평균을 구합니다.
    averages = df.drop(columns=['filename']).mean()

    print("📊 [MCQ Type1 조건별 전체 평균 점수]")
    print("-" * 40)
    print(averages)
    print("-" * 40)

    # 3. 개별적으로 값을 변수에 담고 싶을 경우
    avg_lighten = averages['mcq_type1_lighten_orig']
    avg_orig = averages['mcq_type1_orig_orig']
    avg_darken = averages['mcq_type1_orig_darken']

except FileNotFoundError:
    print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
except Exception as e:
    print(f"❌ 에러 발생: {e}")