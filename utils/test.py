import json
# import numpy as np

score_path = "/data1/joo/pai_bench/result/mcq/cropped/fastcomposer/type1_baseline.json"
with open(score_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

occ_score = []
no_score = []
for item in data:
    string = item['result']
    if string == 'ERROR':
        continue
    id = item['id']
    if id == "014" or id == "027" or id == "028" or id == "029":
        occ_score.append(float(string))
    else:
        no_score.append(float(string))
        
print(f"no avg: {sum(no_score)/len(no_score):.4f}")
print(f"occ avg: {sum(occ_score)/len(occ_score):.4f}")
        
        