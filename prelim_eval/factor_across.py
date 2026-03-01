import pandas as pd
import json
import os
from itertools import combinations

# 설정
metadata_path = '/data1/joo/pai_bench/data/prelim_01/metadata.jsonl'
output_dir = '/data1/joo/pai_bench/results/prelim_01/analysis/factor_pairs/one_same_diff_all'
os.makedirs(output_dir, exist_ok=True)

# 비교할 속성 리스트
factors = ["gender", "ethnicity", "age_group", "hair_color", "facial_expression", "angle", "mustache", "occlusion"]

# Person ID 계산 함수
def get_person_id(img_id):
    return (int(img_id) - 1) // 15

# 1. Metadata 로드
metadata_list = []
with open(metadata_path, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        metadata_list.append(item)

print(f"Total images: {len(metadata_list)}")

# 2. 각 factor별로 처리
for target_factor in factors:
    print(f"\n=== Processing factor: {target_factor} ===")
    
    result_pairs = []
    
    # 모든 이미지 pair 조합 생성
    for i, img1 in enumerate(metadata_list):
        for j, img2 in enumerate(metadata_list):
            if i >= j:  # 중복 제거 (자기 자신과의 비교 및 순서만 다른 pair 제거)
                continue
            
            # 서로 다른 person_id인지 확인 (추가!)
            person_id1 = get_person_id(img1['img_id'])
            person_id2 = get_person_id(img2['img_id'])
            
            if person_id1 == person_id2:  # 같은 person이면 건너뛰기
                continue
            
            # target_factor는 같은지 확인
            if img1[target_factor] != img2[target_factor]:
                continue
            
            # 나머지 factor들이 모두 다른지 확인
            other_factors = [f for f in factors if f != target_factor]
            all_different = all(img1[f] != img2[f] for f in other_factors)
            
            if all_different:
                result_pairs.append({
                    'person_id0': person_id1,
                    'image0': img1['img_id'],
                    'person_id1': person_id2,
                    'image1': img2['img_id'],
                    'same_factor': target_factor,
                    'same_value': img1[target_factor],
                    **{f'img0_{f}': img1[f] for f in other_factors},
                    **{f'img1_{f}': img2[f] for f in other_factors}
                })
    
    # 3. 결과 저장
    if result_pairs:
        df_result = pd.DataFrame(result_pairs)
        save_path = os.path.join(output_dir, f'same_{target_factor}_pairs_across.csv')
        df_result.to_csv(save_path, index=False)
        print(f"  Found {len(result_pairs)} pairs")
        print(f"  Saved to: {save_path}")
        print(f"  Sample:\n{df_result[['person_id0', 'image0', 'person_id1', 'image1', 'same_factor', 'same_value']].head(3)}")
    else:
        print(f"  No pairs found for {target_factor}")

print("\n✅ All factors processed!")