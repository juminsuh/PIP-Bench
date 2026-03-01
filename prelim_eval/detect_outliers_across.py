"""
서로 다른 id를 갖는 이미지들을 대상으로 처리.
IQR을 기준으로 lower / upper outlier 탐지.
"""
import pandas as pd
import numpy as np
import os

similarity_path = '/data1/joo/pai_bench/results/prelim_01/metric/fgis.csv'
output_dir = '/data1/joo/pai_bench/results/prelim_01/analysis/content_outliers/fgis'
os.makedirs(output_dir, exist_ok=True)

df_sim = pd.read_csv(similarity_path)

df_sim['image0'] = df_sim['image0'].apply(lambda x: str(int(x)).zfill(4))
df_sim['image1'] = df_sim['image1'].apply(lambda x: str(int(x)).zfill(4))

def get_id(img_name):
    return (int(img_name) - 1) // 15

df_sim['id0'] = df_sim['image0'].apply(get_id)
df_sim['id1'] = df_sim['image1'].apply(get_id)

df_inter = df_sim[df_sim['id0'] != df_sim['id1']].copy()

def find_inter_outliers(df):
    Q1 = df['fgis_score'].quantile(0.25)
    Q3 = df['fgis_score'].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    print(f"--- Inter-ID Statistics ---")
    print(f"Q1: {Q1:.4f}, Q3: {Q3:.4f}, IQR: {IQR:.4f}")
    print(f"Lower Bound: {lower_bound:.4f}")
    print(f"Upper Bound: {upper_bound:.4f}")
    
    # 이상치 추출
    outliers = df[(df['fgis_score'] < lower_bound) | (df['fgis_score'] > upper_bound)].copy()
    
    # 타입 결정
    if not outliers.empty:
        outliers['outlier_type'] = np.where(
            outliers['fgis_score'] < lower_bound, 'Lower', 'Upper'
        )
    return outliers

# 이상치 추출 실행
outliers_inter = find_inter_outliers(df_inter)

# --- 결과 확인 및 저장 ---
if not outliers_inter.empty:
    # 유사도가 높은(Upper) 순서대로 정렬 (타인인데 닮은 순서)
    outliers_inter = outliers_inter.sort_values(by='fgis_score', ascending=False)
    
    print(f"\n=== Inter-ID Outlier Detection Result ===")
    print(f"Total Inter-ID pairs: {len(df_inter)}")
    print(f"Total outliers found: {len(outliers_inter)}")
    print("\n[Outlier Type Summary]")
    print(outliers_inter['outlier_type'].value_counts())
    
    print("\n[Sample of Inter-ID Outliers (Top 10 Highest Similarity)]")
    print(outliers_inter[['id0', 'image0', 'id1', 'image1', 'fgis_score', 'outlier_type']].head(10))

    # 4. 결과 저장
    save_path = os.path.join(output_dir, "across_outliers.csv")
    outliers_inter.to_csv(save_path, index=False)
    print("\nFile saved: save_path")
else:
    print("No outliers found in Inter-ID pairs.")