# """
# 서로 다른 id를 갖는 이미지들을 대상으로 처리.
# ["facial_expression", "angle", "style"] 3가지 factor.
# 서로 다른 id 사이에서 ["gender", "ethinicity", "age_group"]이 모두 같은 이미지들을 easy pair, 그렇지 않은 나머지 모든 것을 easy pair라고 정의.
# * {factor}_pair_comparison.csv
# - 각 factor의 attribute를 기준으로 해당 이미지들의 within avg 1, within avg 2, cross avg를 측정.

# * {factor}_global_summary.csv
# - 각 factor의 모든 attribute의 값을 within avg 1, within avg 2, cross avg를 기준으로 평균냄. 
# """

# import pandas as pd
# import json
# import os
# from collections import defaultdict

# base_path = "/data1/joo/pai_bench"
# easy_pair_path = f"{base_path}/result/prelim_01/pair/easy_pair.csv"
# metadata_path = f"{base_path}/data/prelim_01/metadata.jsonl"
# mcq_type1_sim_path = f"{base_path}/result/prelim_01/metric/content/mcq_type1_baseline.csv"
# output_dir = f"{base_path}/result/prelim_01/analysis/content_across/mcq_type1_baseline/easy"
# os.makedirs(output_dir, exist_ok=True)

# factors = ['facial_expression', 'angle', 'style']
# error_pairs = [] # "ERROR" 값을 기록하기 위한 리스트

# metadata = []
# with open(metadata_path, 'r', encoding='utf-8') as f:
#     for line in f:
#         try:
#             metadata.append(json.loads(line))
#         except: continue
# df_meta = pd.DataFrame(metadata)
# df_meta['style'] = df_meta['hair_color'] + "_" + df_meta['mustache'].astype(str) + "_" + df_meta['occlusion'].astype(str)

# df_mcq_type1 = pd.read_csv(mcq_type1_sim_path)
# mcq_type1_dict = {}
# for _, row in df_mcq_type1.iterrows():
#     pair = tuple(sorted((str(int(row['image0'])).zfill(4), str(int(row['image1'])).zfill(4))))
#     mcq_type1_dict[pair] = row['mcq_type1_score']

# df_easy = pd.read_csv(easy_pair_path)
# df_easy['img0'] = df_easy['img0'].apply(lambda x: str(int(x)).zfill(4))
# df_easy['img1'] = df_easy['img1'].apply(lambda x: str(int(x)).zfill(4))

# def get_person_start_id(img_id):
#     """이미지 ID를 통해 해당 인물의 시작 ID(0001, 0016...)를 반환"""
#     num = (int(img_id) - 1) // 15
#     return str(num * 15 + 1).zfill(4)

# def run_analysis(factor):
#     print(f"Analyzing factor: {factor}...")
    
#     within_file = f"{base_path}/result/prelim_01/analysis/content_same/mcq_type1_baseline/{factor}_by_id.csv"
#     df_within_ref = pd.read_csv(within_file)
#     within_ref_dict = df_within_ref[df_within_ref['type'] == 'within'].set_index(['id_start', 'key'])['avg_score'].to_dict()

#     # 먼저 각 이미지에 대해 ID를 추가
#     df_easy['id0_start'] = df_easy['img0'].apply(get_person_start_id)
#     df_easy['id1_start'] = df_easy['img1'].apply(get_person_start_id)
    
#     # ID pair만 추출하여 중복 제거
#     id_pairs = df_easy[['id0_start', 'id1_start']].drop_duplicates()
    
#     results = []

#     for _, pair_row in id_pairs.iterrows():
#         start0 = pair_row['id0_start']
#         start1 = pair_row['id1_start']
        
#         # 각 ID에 속한 모든 이미지 가져오기
#         imgs0 = df_meta[df_meta['img_id'].apply(get_person_start_id) == start0]
#         imgs1 = df_meta[df_meta['img_id'].apply(get_person_start_id) == start1]

#         # 두 인물이 공통으로 가지고 있는 속성값(Attribute) 추출
#         attrs0 = set(imgs0[factor].unique())
#         attrs1 = set(imgs1[factor].unique())
#         common_attrs = attrs0.intersection(attrs1)

#         for attr in common_attrs:
#             # 1) img0 내 해당 속성 이미지들의 평균 (파일 참조)
#             score1 = within_ref_dict.get((int(start0), str(attr)), None)
            
#             # 2) img1 내 해당 속성 이미지들의 평균 (파일 참조)
#             score2 = within_ref_dict.get((int(start1), str(attr)), None)

#             # 3) img0과 img1 사이의 해당 속성 이미지들 간의 크로스 유사도 평균
#             cross_imgs0 = imgs0[imgs0[factor] == attr]['img_id'].tolist()
#             cross_imgs1 = imgs1[imgs1[factor] == attr]['img_id'].tolist()
            
#             cross_scores = []
#             for im0 in cross_imgs0:
#                 for im1 in cross_imgs1:
#                     s = mcq_type1_dict.get(tuple(sorted((im0, im1))))
#                     if s is not None:
#                         cross_scores.append(s)
            
#             score3 = sum(cross_scores) / len(cross_scores) if cross_scores else None

#             if score1 is not None and score2 is not None and score3 is not None:
#                 results.append({
#                     'id0_start': start0,
#                     'id1_start': start1,
#                     'attribute': attr,
#                     'within_id0_avg': score1,
#                     'within_id1_avg': score2,
#                     'cross_avg': score3
#                 })

#     # 결과 저장
#     df_res = pd.DataFrame(results)
#     df_res.to_csv(f"{output_dir}/{factor}_pair_comparison.csv", index=False)
    
#     # 전체 평균 계산 및 저장
#     global_avg = df_res[['within_id0_avg', 'within_id1_avg', 'cross_avg']].mean().to_frame().T
#     global_avg['factor'] = factor
#     global_avg.to_csv(f"{output_dir}/{factor}_global_summary.csv", index=False)
    
#     print(f"Saved results for {factor}.")

# # 실행
# for f in factors:
#     run_analysis(f)

import pandas as pd
import json
import os
from collections import defaultdict

base_path = "/data1/joo/pai_bench"
easy_pair_path = f"{base_path}/result/prelim_01/pair/easy_pair.csv"
metadata_path = f"{base_path}/data/prelim_01/metadata.jsonl"
mcq_type1_sim_path = f"{base_path}/result/prelim_01/metric/content/mcq_type1_baseline.csv"
output_dir = f"{base_path}/result/prelim_01/analysis/content_across/mcq_type1_baseline/easy"
os.makedirs(output_dir, exist_ok=True)

factors = ['facial_expression', 'angle', 'style']

metadata = []
with open(metadata_path, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            metadata.append(json.loads(line))
        except: continue
df_meta = pd.DataFrame(metadata)
df_meta['style'] = df_meta['hair_color'] + "_" + df_meta['mustache'].astype(str) + "_" + df_meta['occlusion'].astype(str)

# --- 1. mcq_type1_baseline 로드 및 정제 ---
df_mcq_type1 = pd.read_csv(mcq_type1_sim_path)
mcq_type1_dict = {}
for _, row in df_mcq_type1.iterrows():
    # "ERROR" 값은 딕셔너리에 넣지 않거나 None 처리하여 계산에서 제외되도록 함
    score_val = row['mcq_type1_score']
    if score_val == "ERROR":
        continue
        
    pair = tuple(sorted((str(int(row['image0'])).zfill(4), str(int(row['image1'])).zfill(4))))
    try:
        mcq_type1_dict[pair] = float(score_val)
    except ValueError:
        continue # 숫자로 변환 불가능한 경우 건너뜀

df_easy = pd.read_csv(easy_pair_path)
df_easy['img0'] = df_easy['img0'].apply(lambda x: str(int(x)).zfill(4))
df_easy['img1'] = df_easy['img1'].apply(lambda x: str(int(x)).zfill(4))

def get_person_start_id(img_id):
    num = (int(img_id) - 1) // 15
    return str(num * 15 + 1).zfill(4)

def run_analysis(factor):
    print(f"Analyzing factor: {factor}...")
    
    within_file = f"{base_path}/result/prelim_01/analysis/content_same/mcq_type1_baseline/{factor}_by_id.csv"
    if not os.path.exists(within_file):
        print(f"Skip {factor}: {within_file} not found.")
        return

    df_within_ref = pd.read_csv(within_file)
    # reference 파일 내의 avg_score도 "ERROR"가 섞여 있을 수 있으므로 필터링
    df_within_ref = df_within_ref[df_within_ref['avg_score'] != "ERROR"].copy()
    df_within_ref['avg_score'] = df_within_ref['avg_score'].astype(float)
    
    within_ref_dict = df_within_ref[df_within_ref['type'] == 'within'].set_index(['id_start', 'key'])['avg_score'].to_dict()

    df_easy['id0_start'] = df_easy['img0'].apply(get_person_start_id)
    df_easy['id1_start'] = df_easy['img1'].apply(get_person_start_id)
    
    id_pairs = df_easy[['id0_start', 'id1_start']].drop_duplicates()
    
    results = []

    for _, pair_row in id_pairs.iterrows():
        start0 = pair_row['id0_start']
        start1 = pair_row['id1_start']
        
        imgs0 = df_meta[df_meta['img_id'].apply(get_person_start_id) == start0]
        imgs1 = df_meta[df_meta['img_id'].apply(get_person_start_id) == start1]

        attrs0 = set(imgs0[factor].unique())
        attrs1 = set(imgs1[factor].unique())
        common_attrs = attrs0.intersection(attrs1)

        for attr in common_attrs:
            # 딕셔너리 키 타입(int/str) 일치를 위해 조정
            score1 = within_ref_dict.get((int(start0), str(attr)), None)
            score2 = within_ref_dict.get((int(start1), str(attr)), None)

            cross_imgs0 = imgs0[imgs0[factor] == attr]['img_id'].tolist()
            cross_imgs1 = imgs1[imgs1[factor] == attr]['img_id'].tolist()
            
            cross_scores = []
            for im0 in cross_imgs0:
                for im1 in cross_imgs1:
                    s = mcq_type1_dict.get(tuple(sorted((im0, im1))))
                    # mcq_type1_dict 생성 시 ERROR를 제외했으므로 s는 float이거나 None임
                    if s is not None:
                        cross_scores.append(s)
            
            score3 = sum(cross_scores) / len(cross_scores) if cross_scores else None

            if score1 is not None and score2 is not None and score3 is not None:
                results.append({
                    'id0_start': start0,
                    'id1_start': start1,
                    'attribute': attr,
                    'within_id0_avg': score1,
                    'within_id1_avg': score2,
                    'cross_avg': score3
                })

    if not results:
        print(f"No valid data for {factor}.")
        return

    df_res = pd.DataFrame(results)
    df_res.to_csv(f"{output_dir}/{factor}_pair_comparison.csv", index=False)
    
    global_avg = df_res[['within_id0_avg', 'within_id1_avg', 'cross_avg']].mean().to_frame().T
    global_avg['factor'] = factor
    global_avg.to_csv(f"{output_dir}/{factor}_global_summary.csv", index=False)
    
    print(f"Saved results for {factor}.")

# 실행
for f in factors:
    run_analysis(f)