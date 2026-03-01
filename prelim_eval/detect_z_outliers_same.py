"""
같은 id를 갖는 이미지들을 대상으로 처리.
Z-score를 기준으로 lower / upper outlier 탐지.
중복 pair 제거 포함.
diagonal pair는 결과에서 제외.
"""
import pandas as pd
import numpy as np
import os
from scipy import stats

similarity_path = '/data1/joo/pai_bench/result/prelim_01/metric/content/mcq_type1_coarse.csv'
output_dir = '/data1/joo/pai_bench/result/prelim_01/analysis/content_z_outliers/mcq_type1_coarse'
os.makedirs(output_dir, exist_ok=True)

df_sim = pd.read_csv(similarity_path)

# 1. 이미지 ID를 4자리 문자열(0001, 0002...)로 통일
df_sim['image0'] = df_sim['image0'].apply(lambda x: str(int(x)).zfill(4))
df_sim['image1'] = df_sim['image1'].apply(lambda x: str(int(x)).zfill(4))

# 1-1. diagonal pair 식별
df_sim['is_diagonal'] = df_sim['image0'] == df_sim['image1']

# 1-2. 중복 pair 제거 (image0, image1 순서만 다른 경우)
# 각 행에서 작은 값을 min_img, 큰 값을 max_img로 설정
df_sim['min_img'] = df_sim[['image0', 'image1']].min(axis=1)
df_sim['max_img'] = df_sim[['image0', 'image1']].max(axis=1)

# 중복 제거
df_sim = df_sim.drop_duplicates(subset=['min_img', 'max_img'], keep='first')

print(f"Total pairs after deduplication: {len(df_sim)}")

def get_id(img_name):
    return (int(img_name) - 1) // 15

df_sim['id0'] = df_sim['image0'].apply(get_id)
df_sim['id1'] = df_sim['image1'].apply(get_id)

# 2. 같은 ID 내의 Pair만 필터링
df_intra = df_sim[df_sim['id0'] == df_sim['id1']].copy()
df_intra.rename(columns={'id0': 'person_id'}, inplace=True)

def find_id_specific_z_outliers(group):
    # diagonal 제외하고 평균/표준편차 계산
    non_diagonal = group[~group['is_diagonal']]
    
    if len(non_diagonal) < 2:
        return pd.DataFrame()
    
    mean = non_diagonal['mcq_type1_coarse_score'].mean()
    std = non_diagonal['mcq_type1_coarse_score'].std()
    
    if std == 0:
        return pd.DataFrame()
    
    # 전체 그룹에 대해 Z-score 계산 (diagonal 제외 통계 사용)
    group['z_score'] = (group['mcq_type1_coarse_score'] - mean) / std
    
    # 이상치 추출 (diagonal 제외!)
    outliers = group[
        ((group['z_score'] < -2.0) | (group['z_score'] > 2.0)) & 
        (~group['is_diagonal'])
    ].copy()
    
    if not outliers.empty:
        outliers['outlier_type'] = np.where(
            outliers['z_score'] < -2.0, 'Lower', 'Upper'
        )
    
    return outliers

# 3. ID별로 그룹화하여 이상치 추출 및 타입 지정
outliers_by_id = df_intra.groupby('person_id', group_keys=False).apply(find_id_specific_z_outliers)

# --- 결과 확인 및 저장 ---
if not outliers_by_id.empty:
    # 정렬
    outliers_by_id = outliers_by_id.sort_values(by=['person_id', 'z_score'])
    
    print(f"\n=== Intra-ID Z-score Outlier Detection Result ===")
    print(f"Total outliers found: {len(outliers_by_id)}")
    print(f"Diagonal pairs excluded: {(~outliers_by_id['is_diagonal']).sum()} (should be all)")
    print("\n[Outlier Type Summary]")
    print(outliers_by_id['outlier_type'].value_counts())
    
    print("\n[Sample of Outliers]")
    print(outliers_by_id[['person_id', 'image0', 'image1', 'mcq_type1_coarse_score', 'z_score', 'outlier_type']].head(10))

    # 4. 결과 저장
    output_path = os.path.join(output_dir, "same_outliers.csv")
    outliers_by_id.to_csv(output_path, index=False)
    print(f"\nFile saved: {output_path}")
else:
    print("No outliers found.")