import pandas as pd
import numpy as np
import os

# --- 설정 ---
similarity_path = '/data1/joo/pai_bench/result/prelim_01/metric/content/mcq_type1.csv'
output_dir = '/data1/joo/pai_bench/result/prelim_01/analysis/outlier_mad/mcq_type1'
os.makedirs(output_dir, exist_ok=True)

# 1. 데이터 로드 (dtype 명시로 0001 포맷 유지)
df_sim = pd.read_csv(similarity_path, dtype={'image0': str, 'image1': str})

# 1-1. diagonal pair 식별 및 중복 제거 (기존 로직 유지)
df_sim['is_diagonal'] = df_sim['image0'] == df_sim['image1']
df_sim['min_img'] = df_sim[['image0', 'image1']].min(axis=1)
df_sim['max_img'] = df_sim[['image0', 'image1']].max(axis=1)
df_sim = df_sim.drop_duplicates(subset=['min_img', 'max_img'], keep='first')

# 1-2. Person ID 생성 (기존 로직: 15개 단위로 그룹핑)
def get_id(img_name):
    return (int(img_name) - 1) // 15

df_sim['person_id'] = df_sim['image0'].apply(get_id)

# 2. 같은 ID 내의 Pair만 필터링 (Intra-ID 분석)
df_intra = df_sim[df_sim['person_id'] == df_sim['image1'].apply(get_id)].copy()

def find_id_specific_mad_outliers(group):
    """
    Modified Z-score (MAD 기반)를 사용하여 하한 이상치 탐지
    """
    # diagonal 제외 통계 계산
    non_diagonal = group[~group['is_diagonal']]
    
    if len(non_diagonal) < 2:
        return pd.DataFrame()
    
    scores = non_diagonal['mcq_type1_score']
    median = scores.median()
    # MAD 계산: 중앙값으로부터의 절대 편차들의 중앙값
    mad = np.median(np.abs(scores - median))
    
    if mad == 0:
        # 모든 데이터가 같을 경우 처리 (이상치 없음)
        return pd.DataFrame()
    
    # Modified Z-score 계산 (상수 0.6745는 정규분포 정렬용)
    # 수식: 0.6745 * (x - median) / MAD
    group['modified_z'] = 0.6745 * (group['mcq_type1_score'] - median) / mad
    
    # Lower Outlier 탐지 (일반적인 임계값 3.5 사용)
    # 0.3~0.7 사이의 값이 비정상적으로 작다면 여기서 걸러집니다.
    lower_outliers = group[
        (group['modified_z'] < -3.5) & 
        (~group['is_diagonal'])
    ].copy()
    
    if not lower_outliers.empty:
        lower_outliers['outlier_type'] = 'Lower'
        # 분석을 위해 해당 그룹의 통계치 추가 기록
        lower_outliers['group_median'] = median
        lower_outliers['group_mad'] = mad
        
    return lower_outliers

# 3. ID별 그룹화 및 이상치 탐지 수행
outliers_by_id = df_intra.groupby('person_id', group_keys=False).apply(find_id_specific_mad_outliers)

# --- 결과 확인 및 저장 ---
if not outliers_by_id.empty:
    # 정렬 (가장 점수가 낮은 이상치부터)
    outliers_by_id = outliers_by_id.sort_values(by=['person_id', 'mcq_type1_score'])
    
    print(f"\n=== Intra-ID MAD Outlier Detection Result ===")
    print(f"Total lower outliers found: {len(outliers_by_id)}")
    
    # 상위 10개 출력
    cols_to_show = ['person_id', 'image0', 'image1', 'mcq_type1_score', 'modified_z', 'group_median', 'outlier_type']
    print(outliers_by_id[cols_to_show].head(10))

    # 4. 결과 저장
    output_path = os.path.join(output_dir, "lower_outliers_analysis.csv")
    outliers_by_id.to_csv(output_path, index=False)
    print(f"\n✅ 분석 완료! 파일 저장 위치: {output_path}")
else:
    print("❌ 이상치가 발견되지 않았습니다.")