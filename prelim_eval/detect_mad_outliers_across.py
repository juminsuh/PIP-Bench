import pandas as pd
import numpy as np
import os

# --- 설정 ---
similarity_path = '/data1/joo/pai_bench/result/prelim_01/metric/content/mcq_type1.csv'
output_dir = '/data1/joo/pai_bench/result/prelim_01/analysis/outlier_mad/mcq_type1'
os.makedirs(output_dir, exist_ok=True)

# 1. 데이터 로드 (dtype 명시로 0001 포맷 유지)
# 'score' 열 이름이 mcq_type1_score인지 확인 필요 (이전 대화 기준 mcq_type1_score 사용)
df_sim = pd.read_csv(similarity_path, dtype={'image0': str, 'image1': str})

# 1-1. 중복 pair 제거 (image0, image1 순서만 다른 경우 하나로 합침)
df_sim['min_img'] = df_sim[['image0', 'image1']].min(axis=1)
df_sim['max_img'] = df_sim[['image0', 'image1']].max(axis=1)
df_sim = df_sim.drop_duplicates(subset=['min_img', 'max_img'], keep='first')

# 1-2. ID 추출 함수 및 ID 부여
def get_id(img_name):
    return (int(img_name) - 1) // 15

df_sim['id0'] = df_sim['image0'].apply(get_id)
df_sim['id1'] = df_sim['image1'].apply(get_id)

# 2. 서로 다른 ID 간의 Pair만 필터링 (Inter-ID 분석)
# image0의 주인과 image1의 주인이 다른 데이터만 추출
df_inter = df_sim[df_sim['id0'] != df_sim['id1']].copy()

print(f"🚀 분석 대상 (Inter-ID Pairs): {len(df_inter)}개")

# 3. 전체 Inter-ID 분포에 대한 MAD 기반 이상치 탐지
def find_upper_outliers_mad(df, score_col='mcq_type1_score', threshold=3.5):
    if df.empty:
        return pd.DataFrame()
    
    scores = df[score_col]
    median = scores.median()
    # MAD (중앙값 절대 편차) 계산
    mad = np.median(np.abs(scores - median))
    
    print(f"📊 Global Statistics | Median: {median:.4f}, MAD: {mad:.4f}")
    
    if mad == 0:
        print("⚠️ MAD가 0입니다. 모든 데이터 점수가 동일하여 이상치를 탐지할 수 없습니다.")
        return pd.DataFrame()
    
    # Modified Z-score 계산
    df['modified_z'] = 0.6745 * (df[score_col] - median) / mad
    
    # Upper Outlier (타인인데 점수가 임계값보다 높은 경우)
    upper_outliers = df[df['modified_z'] > threshold].copy()
    
    if not upper_outliers.empty:
        upper_outliers['outlier_type'] = 'Upper'
        upper_outliers['global_median'] = median
        upper_outliers['global_mad'] = mad
        
    return upper_outliers

# 4. 실행
# 데이터프레임의 실제 점수 컬럼명에 맞춰 score_col 인자를 확인하세요.
outliers_result = find_upper_outliers_mad(df_inter, score_col='mcq_type1_score')

# --- 결과 확인 및 저장 ---
if not outliers_result.empty:
    # 점수가 높은 순서대로 정렬 (심각한 이상치부터 확인)
    outliers_result = outliers_result.sort_values(by='mcq_type1_score', ascending=False)
    
    print(f"\n✨ Total Upper Outliers Found: {len(outliers_result)}")
    
    # 주요 컬럼 위주로 출력
    cols_to_show = ['image0', 'image1', 'id0', 'id1', 'mcq_type1_score', 'modified_z']
    print("\n[Top 10 Severe Outliers]")
    print(outliers_result[cols_to_show].head(10))

    # 5. 결과 저장
    output_path = os.path.join(output_dir, "inter_upper_outliers_analysis.csv")
    outliers_result.to_csv(output_path, index=False)
    print(f"\n✅ 분석 파일이 저장되었습니다: {output_path}")
else:
    print("\n❌ 이상치가 발견되지 않았습니다. 모든 타인 쌍의 점수가 정상 범위 내에 있습니다.")