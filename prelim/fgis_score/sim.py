import pandas as pd
import numpy as np
import pickle
import os
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity

def load_all_embeddings(input_dir):
    """디렉토리 내의 모든 .pkl 파일을 읽어 하나의 DataFrame으로 합칩니다."""
    all_dfs = []
    print(f"📂 데이터 로드 중: {input_dir}")
    
    files = [f for f in os.listdir(input_dir) if f.endswith('.pkl')]
    for file in files:
        with open(os.path.join(input_dir, file), 'rb') as f:
            data = pickle.load(f)
            # 리스트 형태면 DF로 변환, 이미 DF면 그대로 사용
            df_temp = pd.DataFrame(data)
            all_dfs.append(df_temp)
    
    if not all_dfs:
        return None
        
    combined_df = pd.concat(all_dfs, ignore_index=True)
    # 이미지 ID 포맷 통일 (0001, 0002...)
    combined_df['image_id'] = combined_df['image_id'].apply(lambda x: str(int(x)).zfill(4))
    return combined_df

def compute_all_pairs_similarity(df, output_path):
    """모든 이미지 조합에 대해 11개 부위별 유사도 평균을 계산합니다."""
    # 요청하신 11개 얼굴 부위 컬럼
    feature_cols = [1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13]
    image_ids = df['image_id'].tolist()
    
    # 조회를 빠르게 하기 위해 ID를 인덱스로 설정
    df_indexed = df.set_index('image_id')
    
    results = []
    # 고유한 쌍(Pair) 생성 (N * (N-1) / 2)
    pairs = list(combinations(image_ids, 2))
    total_pairs = len(pairs)
    
    print(f"🚀 총 {len(image_ids)}개의 이미지에서 {total_pairs}개의 쌍을 계산합니다.")

    for i, (id1, id2) in enumerate(pairs):
        row1 = df_indexed.loc[id1]
        row2 = df_indexed.loc[id2]
        
        feature_similarities = []
        
        for col in feature_cols:
            emb1 = row1.get(col)
            emb2 = row2.get(col)
            
            # 두 이미지 모두 해당 부위 임베딩이 존재하는 경우만
            if emb1 is not None and emb2 is not None:
                # 데이터가 비어있지 않은지 확인
                if isinstance(emb1, (list, np.ndarray)) and len(emb1) > 0:
                    vec1 = np.array(emb1).reshape(1, -1)
                    vec2 = np.array(emb2).reshape(1, -1)
                    
                    sim = cosine_similarity(vec1, vec2)[0][0]
                    feature_similarities.append(sim)
        
        # 11개 부위의 평균 유사도 산출
        if len(feature_similarities) > 0:
            avg_sim = np.mean(feature_similarities)
            results.append({
                'img0': id1,
                'img1': id2,
                'fgis_sim': avg_sim,
                'features_cnt': len(feature_similarities)
            })
            
        # 진행 상황 출력 (1000단위)
        if (i + 1) % 1000 == 0:
            print(f"⏳ 진행 중... ({i + 1}/{total_pairs})")

    # 결과 저장
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)
    return result_df

def main():
    input_dir = "/data1/joo/pai_bench/data/prelim_01/fgis/content/emb"
    output_dir = "/data1/joo/pai_bench/results/prelim_01/metric"
    output_file = os.path.join(output_dir, "fgis.csv")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 모든 데이터 로드
    full_df = load_all_embeddings(input_dir)
    
    if full_df is not None:
        # 2. 모든 조합 유사도 계산
        final_results = compute_all_pairs_similarity(full_df, output_file)
        
        print(f"✅ 모든 계산이 완료되었습니다.")

if __name__ == "__main__":
    main()