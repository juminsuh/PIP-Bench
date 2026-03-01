"""
같은 id를 갖는 이미지들을 대상으로 처리.
IQR을 기준으로 lower / upper outlier 탐지.
"""
import pandas as pd
import numpy as np
import os

similarity_path = '/data1/joo/pai_bench/results/prelim_01/metric/content/fgis.csv'
output_dir = '/data1/joo/pai_bench/results/prelim_01/analysis/content_outliers/fgis'
os.makedirs(output_dir, exist_ok=True)

df_sim = pd.read_csv(similarity_path)

# 1. 이미지 ID를 4자리 문자열(0001, 0002...)로 통일
df_sim['image0'] = df_sim['image0'].apply(lambda x: str(int(x)).zfill(4))
df_sim['image1'] = df_sim['image1'].apply(lambda x: str(int(x)).zfill(4))

def get_id(img_name):
    return (int(img_name) - 1) // 15

df_sim['id0'] = df_sim['image0'].apply(get_id)
df_sim['id1'] = df_sim['image1'].apply(get_id)

# 2. 같은 ID 내의 Pair만 필터링
df_intra = df_sim[df_sim['id0'] == df_sim['id1']].copy()
df_intra.rename(columns={'id0': 'person_id'}, inplace=True)

def find_id_specific_outliers(group):
    """각 ID 그룹별로 IQR을 계산하고 이상치 타입을 명시합니다."""
    Q1 = group['fgis_score'].quantile(0.25)
    Q3 = group['fgis_score'].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # 이상치 추출
    outliers = group[(group['fgis_score'] < lower_bound) | (group['fgis_score'] > upper_bound)].copy()
    
    # 타입 결정 (벡터 연산으로 조건 할당)
    if not outliers.empty:
        outliers['outlier_type'] = np.where(
            outliers['fgis_score'] < lower_bound, 'Lower', 'Upper'
        )
    
    return outliers

# 3. ID별로 그룹화하여 이상치 추출 및 타입 지정
outliers_by_id = df_intra.groupby('person_id', group_keys=False).apply(find_id_specific_outliers)

# --- 결과 확인 및 저장 ---
if not outliers_by_id.empty:
    # 정렬
    outliers_by_id = outliers_by_id.sort_values(by=['person_id', 'fgis_score'])
    
    print(f"=== Intra-ID Outlier Detection Result ===")
    print(f"Total outliers found: {len(outliers_by_id)}")
    print("\n[Outlier Type Summary]")
    print(outliers_by_id['outlier_type'].value_counts()) # Lower/Upper 개수 요약 출력
    
    print("\n[Sample of Outliers]")
    print(outliers_by_id[['person_id', 'image0', 'image1', 'fgis_score', 'outlier_type']].head(10))

    # 4. 결과 저장
    output_path = os.path.join(output_dir, "same_outliers.csv")
    outliers_by_id.to_csv(output_path, index=False)
    print(f"\nFile saved: {output_path}")
else:
    print("No outliers found.")