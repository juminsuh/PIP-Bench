import pandas as pd
import os

# 설정
pairs_dir = '/data1/joo/pai_bench/results/prelim_01/analysis/factor_pairs/one_diff_same_all'
similarity_path = '/data1/joo/pai_bench/results/prelim_01/metric/content/fgis.csv'
output_dir = '/data1/joo/pai_bench/results/prelim_01/analysis/factor_pairs/one_diff_same_all/fgis'
os.makedirs(output_dir, exist_ok=True)

# 비교할 속성 리스트
factors = ["gender", "ethnicity", "age_group", "hair_color", "facial_expression", "angle", "mustache", "occlusion"]

# 1. 유사도 데이터 로드
df_sim = pd.read_csv(similarity_path)

# 이미지 ID를 4자리 문자열로 통일
df_sim['image0'] = df_sim['image0'].apply(lambda x: str(int(x)).zfill(4))
df_sim['image1'] = df_sim['image1'].apply(lambda x: str(int(x)).zfill(4))

# 정규화된 pair를 키로 하는 딕셔너리 생성 (작은 값, 큰 값 순으로 정렬)
sim_dict = {}
for _, row in df_sim.iterrows():
    img0, img1 = row['image0'], row['image1']
    score = row['fgis_score']
    
    # 항상 작은 값을 먼저, 큰 값을 나중에 (정규화)
    key = tuple(sorted([img0, img1]))
    sim_dict[key] = score

print(f"Loaded {len(sim_dict)} unique similarity pairs")

# 2. 각 factor별로 처리
results_summary = []

for target_factor in factors:
    pairs_file = os.path.join(pairs_dir, f'diff_{target_factor}_pairs_same.csv')
    
    # 파일이 존재하는지 확인
    if not os.path.exists(pairs_file):
        print(f"\n⚠️  File not found: {pairs_file}")
        continue
    
    print(f"\n=== Processing factor: {target_factor} ===")
    
    # Pair 데이터 로드
    df_pairs = pd.read_csv(pairs_file)
    
    # 각 pair의 유사도 추출
    similarities = []
    missing_count = 0
    
    for _, row in df_pairs.iterrows():
        img0 = str(row['image0']).zfill(4)
        img1 = str(row['image1']).zfill(4)
        
        # 정규화된 키로 검색 (작은 값, 큰 값 순)
        key = tuple(sorted([img0, img1]))
        score = sim_dict.get(key)
        
        if score is not None:
            similarities.append(score)
        else:
            missing_count += 1
            print(f"    ⚠️  Missing pair: {img0} - {img1}")
    
    # 평균 계산
    if similarities:
        avg_similarity = sum(similarities) / len(similarities)
        results_summary.append({
            'factor': target_factor,
            'num_pairs': len(df_pairs),
            'num_matched': len(similarities),
            'num_missing': missing_count,
            'avg_similarity': avg_similarity
        })
        
        print(f"  Total pairs: {len(df_pairs)}")
        print(f"  Matched pairs: {len(similarities)}")
        print(f"  Missing pairs: {missing_count}")
        print(f"  Average similarity: {avg_similarity:.4f}")
    else:
        print(f"  No matching similarities found!")

# 3. 결과 요약 저장
if results_summary:
    df_summary = pd.DataFrame(results_summary)
    df_summary = df_summary.sort_values('avg_similarity', ascending=False)
    
    summary_path = os.path.join(output_dir, 'similarity_summary_by_factor.csv')
    df_summary.to_csv(summary_path, index=False)
    
    print("\n" + "="*60)
    print("=== Summary: Average Similarity by Factor (Same ID) ===")
    print("="*60)
    print(df_summary.to_string(index=False))
    print(f"\n✅ Summary saved to: {summary_path}")
else:
    print("\n⚠️  No results to summarize")