"""
같은 id를 갖는 이미지들을 대상으로 처리.
["facial_expression", "angle", "style"] 3가지 factor.

* {factor}_by_id.csv
1) within: 같은 id 내의 각 factor의 각 attribute를 기준으로 같은 이미지들 사이의 유사도의 평균을 측정. 
2) across: 같은 id 내의 각 factor의 각 attribute를 기준으로 다른 이미지들 사이의 유사도의 평균을 측정.

* {factor}_global_summary.csv
각 factor의 각 attribute를 기준으로 총 13개의 id에 대한 평균을 측정. (개별값들의 평균)
"""

import pandas as pd
import json
import os
from collections import defaultdict
from itertools import combinations

# --- 설정 ---
metadata_path = '/data1/joo/pai_bench/data/prelim_01/metadata.jsonl'
# similarity_path = '/data1/joo/pai_bench/results/prelim_01/metric/content/arcface.csv'
similarity_path = '/data1/joo/pai_bench/result/prelim_01/metric/content/mcq_type1_baseline.csv'
output_dir = '/data1/joo/pai_bench/result/prelim_01/analysis/content_same/mcq_type1_baseline'
os.makedirs(output_dir, exist_ok=True)

# 1. 메타데이터 및 유사도 로드
metadata = []
with open(metadata_path, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            line = line.strip()
            if not line: continue
            metadata.append(json.loads(line))
        except: continue

df_meta = pd.DataFrame(metadata)
df_sim = pd.read_csv(similarity_path)

df_sim_clean = df_sim[df_sim['mcq_type1_score'] != "ERROR"].copy()
df_sim_clean['mcq_type1_score'] = df_sim_clean['mcq_type1_score'].astype(float)

# 2. sim_dict 생성 (문자열 인덱스 매핑)
sim_dict = {
    tuple(sorted((str(int(r[0])).zfill(4), str(int(r[1])).zfill(4)))): r[2] 
    for r in df_sim_clean[['image0', 'image1', 'mcq_type1_score']].values
}

# sim_dict = {tuple(sorted((str(int(r[0])).zfill(4), str(int(r[1])).zfill(4)))): r[2] 
#             for r in df_sim[['image0', 'image1','mcq_type1_score']].values}

def extract_pair(id_group, factor):
    within_scores = defaultdict(list)
    across_scores = defaultdict(list)
    style_elements = ['hair_color', 'mustache', 'occlusion']
    
    # 미리 매핑하여 속도 향상
    if factor == 'style':
        attr_map = {row['img_id']: tuple(row[style_elements]) for _, row in id_group.iterrows()}
    else:
        attr_map = {row['img_id']: row[factor] for _, row in id_group.iterrows()}
    
    img_ids = id_group['img_id'].tolist()
    for img1, img2 in combinations(img_ids, 2):
        score = sim_dict.get(tuple(sorted((img1, img2))))
        if score is None: continue
        
        attr1, attr2 = attr_map[img1], attr_map[img2]
        
        if attr1 == attr2:
            key = "_".join(map(str, attr1)) if factor == 'style' else attr1
            within_scores[key].append(score)
        else:
            key1 = "_".join(map(str, attr1)) if factor == 'style' else str(attr1)
            key2 = "_".join(map(str, attr2)) if factor == 'style' else str(attr2)
            pair_key = " vs ".join(sorted((key1, key2))) # CSV 저장을 위해 문자열화
            across_scores[pair_key].append(score)
            
    return within_scores, across_scores

def analyze_factor(factor):
    print(f"\nAnalyzing Factor: {factor}...")
    
    id_results = [] # ID별 결과를 담을 리스트
    total_within = defaultdict(list)
    total_across = defaultdict(list)
    
    for i in range(0, len(df_meta), 15):
        id_group = df_meta.iloc[i:i+15]
        start_id = id_group.iloc[0]['img_id']
        within, across = extract_pair(id_group, factor)
        
        # 1. ID별 데이터 수집
        for attr, scores in within.items():
            avg = sum(scores)/len(scores)
            id_results.append({'id_start': start_id, 'type': 'within', 'key': attr, 'avg_score': avg, 'count': len(scores)})
            total_within[attr].extend(scores)
        for pair_key, scores in across.items():
            avg = sum(scores)/len(scores)
            id_results.append({'id_start': start_id, 'type': 'across', 'key': pair_key, 'avg_score': avg, 'count': len(scores)})
            total_across[pair_key].extend(scores)

    # 2. 전체(Global) 데이터 요약
    global_results = []
    for attr, scores in total_within.items():
        global_results.append({'type': 'within', 'key': attr, 'avg_score': sum(scores)/len(scores), 'total_count': len(scores)})
    for pair_key, scores in total_across.items():
        global_results.append({'type': 'across', 'key': pair_key, 'avg_score': sum(scores)/len(scores), 'total_count': len(scores)})

    # --- CSV 저장 ---
    df_id_final = pd.DataFrame(id_results)
    df_global_final = pd.DataFrame(global_results)
    
    df_id_final.to_csv(f"{output_dir}/{factor}_by_id.csv", index=False)
    df_global_final.to_csv(f"{output_dir}/{factor}_global_avg.csv", index=False)
    
    print(f"Done! Files saved in: {output_dir}")
    return df_global_final # 결과 확인용 반환

# 실행
analyze_factor('style')
analyze_factor('facial_expression')
analyze_factor('angle')