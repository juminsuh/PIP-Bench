import pandas as pd
import json
import os

# 1. 데이터 로드
outliers_df = pd.read_csv('/data1/joo/pai_bench/result/prelim_01/analysis/content_z_outliers/mcq_type1/across_outliers.csv')
metadata_path = '/data1/joo/pai_bench/data/prelim_01/metadata.jsonl'

# 2. upper Outlier만 필터링
upper_outliers = outliers_df[outliers_df['outlier_type'] == 'Upper'].copy()

# 3. Metadata 로드 (img_id를 키로 하는 딕셔너리 생성)
meta_dict = {}
with open(metadata_path, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        # ID 포맷 통일 (예: 1 -> "0001")
        img_id = str(item['img_id']).zfill(4)
        meta_dict[img_id] = item

# 4. 비교할 속성 리스트
factors = ['gender','ethnicity', 'age_group', 'hair_color', 'facial_expression', 'angle', 'mustache', 'occlusion']

# 5. 비교 로직 함수
def analyze_differences(row):
    img0, img1 = str(row['image0']).zfill(4), str(row['image1']).zfill(4)
    
    # 메타데이터 가져오기
    meta0 = meta_dict.get(img0)
    meta1 = meta_dict.get(img1)
    
    if not meta0 or not meta1:
        return "metadata_missing"

    diffs = []
    for factor in factors:
        val0 = meta0.get(factor)
        val1 = meta1.get(factor)
        
        # 값이 다를 경우 리스트에 추가 (예: "angle(Left vs Right)")
        if val0 != val1:
            diffs.append(f"{factor}({val0} vs {val1})")
            
    return ", ".join(diffs) if diffs else "No differences found"

# 6. 분석 적용 및 결과 저장
upper_outliers['diff_factors'] = upper_outliers.apply(analyze_differences, axis=1)

# 결과 확인 (상위 5개)
print(upper_outliers[['image0', 'image1', 'mcq_type1_score', 'diff_factors']].head())

# .csv 파일로 저장
save_dir = "/data1/joo/pai_bench/result/prelim_01/analysis/outlier_z_meta/mcq_type1"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'upper_outliers_analysis.csv')

upper_outliers.to_csv(save_path, index=False)
print(f"\n✅ 분석 결과가 {save_path}로 저장되었습니다.")