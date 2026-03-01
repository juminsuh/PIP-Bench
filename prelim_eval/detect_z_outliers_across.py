"""
서로 다른 id를 갖는 이미지들을 대상으로 처리.
Z-score를 기준으로 lower / upper outlier 탐지.
중복 pair 제거 포함.
"""
import pandas as pd
import numpy as np
import os
from scipy import stats

similarity_path = '/data1/joo/pai_bench/result/prelim_01/metric/content/mcq_type1_fine.csv'
output_dir = '/data1/joo/pai_bench/result/prelim_01/analysis/content_z_outliers/mcq_type1_fine'
os.makedirs(output_dir, exist_ok=True)

df_sim = pd.read_csv(similarity_path)

# 1. 이미지 ID를 4자리 문자열(0001, 0002...)로 통일
df_sim['image0'] = df_sim['image0'].apply(lambda x: str(int(x)).zfill(4))
df_sim['image1'] = df_sim['image1'].apply(lambda x: str(int(x)).zfill(4))

# 1-1. diagonal pair 식별
df_sim['is_diagonal'] = df_sim['image0'] == df_sim['image1']

# 1-2. 중복 pair 제거 (image0, image1 순서만 다른 경우)
df_sim['min_img'] = df_sim[['image0', 'image1']].min(axis=1)
df_sim['max_img'] = df_sim[['image0', 'image1']].max(axis=1)
df_sim = df_sim.drop_duplicates(subset=['min_img', 'max_img'], keep='first')

print(f"Total pairs after deduplication: {len(df_sim)}")

def get_id(img_name):
    return (int(img_name) - 1) // 15

df_sim['id0'] = df_sim['image0'].apply(get_id)
df_sim['id1'] = df_sim['image1'].apply(get_id)

# 2. 서로 다른 ID 간의 Pair만 필터링
df_inter = df_sim[df_sim['id0'] != df_sim['id1']].copy()

print(f"Total inter-ID pairs: {len(df_inter)}")

# 3. Z-score 계산
mean = df_inter['mcq_type1_fine_score'].mean()
std = df_inter['mcq_type1_fine_score'].std()

print(f"\n--- Inter-ID Statistics ---")
print(f"Mean: {mean:.4f}")
print(f"Std: {std:.4f}")

if std == 0:
    print("Standard deviation is 0. Cannot calculate Z-scores.")
else:
    df_inter['z_score'] = (df_inter['mcq_type1_fine_score'] - mean) / std
    
    # 4. 이상치 추출 (Z < -2.0 or Z > 2.0)
    outliers_inter = df_inter[
        (df_inter['z_score'] < -2.0) | (df_inter['z_score'] > 2.0)
    ].copy()
    
    # 5. 타입 결정
    if not outliers_inter.empty:
        outliers_inter['outlier_type'] = np.where(
            outliers_inter['z_score'] < -2.0, 'Lower', 'Upper'
        )
        
        # 유사도가 높은(Upper) 순서대로 정렬
        outliers_inter = outliers_inter.sort_values(by='mcq_type1_fine_score', ascending=False)
        
        print(f"\n=== Inter-ID Z-score Outlier Detection Result ===")
        print(f"Total outliers found: {len(outliers_inter)}")
        print("\n[Outlier Type Summary]")
        print(outliers_inter['outlier_type'].value_counts())
        
        print("\n[Sample of Inter-ID Outliers (Top 10 Highest Similarity)]")
        print(outliers_inter[['id0', 'image0', 'id1', 'image1', 'mcq_type1_fine_score', 'z_score', 'outlier_type']].head(10))
        
        # 6. 결과 저장
        save_path = os.path.join(output_dir, "across_outliers.csv")
        outliers_inter.to_csv(save_path, index=False)
        print(f"\nFile saved: {save_path}")
    else:
        print("\nNo outliers found in Inter-ID pairs.")