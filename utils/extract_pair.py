import pandas as pd
import json
from itertools import combinations

metadata_path = '/data1/joo/pai_bench/data/prelim_01/metadata.jsonl'
metadata = []
with open(metadata_path, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            line = line.strip()
            if not line: continue
            metadata.append(json.loads(line))
        except: continue

df_meta = pd.DataFrame(metadata)

img_info = {}
for _, row in df_meta.iterrows():
    img_info[row['img_id']] = {
        'id': (int(row['img_id']) - 1) // 15,
        'demo': (row['gender'], row['ethnicity'], row['age_group'])
    }

img_ids = sorted(img_info.keys())

same_demo_pairs = []
diff_demo_pairs = []

print("Pair 분석 중...")
for img1, img2 in combinations(img_ids, 2):
    info1 = img_info[img1]
    info2 = img_info[img2]
    
    if info1['id'] != info2['id']:
        if info1['demo'] == info2['demo']:
            same_demo_pairs.append((img1, img2))
        else:
            diff_demo_pairs.append((img1, img2))

# 4. 결과 출력
print("\n" + "="*50)
print(f" 분석 결과 (Inter-ID Only)")
print("="*50)
print(f"1. Demographic 모두 일치하는 Pair 수: {len(same_demo_pairs)}")
print(f"2. Demographic 하나라도 다른 Pair 수: {len(diff_demo_pairs)}")
print("="*50)

# 샘플 출력 (각 10개씩)
print("\n[Same Demographic Pairs Sample (First 10)]")
for p in same_demo_pairs[:10]:
    print(f"{p[0]} <-> {p[1]} | Attributes: {img_info[p[0]]['demo']}")

print("\n[Different Demographic Pairs Sample (First 10)]")
for p in diff_demo_pairs[:10]:
    attr1 = img_info[p[0]]['demo']
    attr2 = img_info[p[1]]['demo']
    print(f"{p[0]} <-> {p[1]} | {attr1} vs {attr2}")

# 5. (선택 사항) 결과 저장
pd.DataFrame(same_demo_pairs, columns=['img0', 'img1']).to_csv('/data1/joo/pai_bench/results/prelim_01/analysis/pair/diff_pair.csv', index=False)
pd.DataFrame(diff_demo_pairs, columns=['img0', 'img1']).to_csv('/data1/joo/pai_bench/results/prelim_01/analysis/easy_pair.csv', index=False)
