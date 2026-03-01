import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def load_embeddings_from_dir(input_dir):
    """디렉토리 내의 모든 .pkl 파일을 읽어 image_id를 인덱스로 하는 DataFrame을 반환합니다."""
    all_dfs = []
    if not os.path.exists(input_dir):
        print(f"⚠️ 경로를 찾을 수 없습니다: {input_dir}")
        return None
        
    files = [f for f in os.listdir(input_dir) if f.endswith('.pkl')]
    for file in files:
        with open(os.path.join(input_dir, file), 'rb') as f:
            data = pickle.load(f)
            df_temp = pd.DataFrame(data)
            all_dfs.append(df_temp)
    
    if not all_dfs:
        return None
        
    combined_df = pd.concat(all_dfs, ignore_index=True)
    # 이미지 ID 포맷 통일 및 인덱스 설정
    combined_df['image_id'] = combined_df['image_id'].apply(lambda x: str(int(x)).zfill(4))
    return combined_df.set_index('image_id')

def calculate_pair_similarity(row1, row2, feature_cols):
    """두 행 사이의 11개 부위별 유사도 평균을 계산합니다."""
    feature_similarities = []
    
    for col in feature_cols:
        emb1 = row1.get(col)
        emb2 = row2.get(col)
        
        if emb1 is not None and emb2 is not None:
            # 데이터 유효성 검사 (리스트나 넘파이 배열이고 비어있지 않아야 함)
            if isinstance(emb1, (list, np.ndarray)) and len(emb1) > 0 and \
               isinstance(emb2, (list, np.ndarray)) and len(emb2) > 0:
                
                vec1 = np.array(emb1).reshape(1, -1)
                vec2 = np.array(emb2).reshape(1, -1)
                
                sim = cosine_similarity(vec1, vec2)[0][0]
                feature_similarities.append(sim)
    
    return np.mean(feature_similarities) if feature_similarities else None

def main():
    # 1. 경로 설정
    base_path = Path("/data1/joo/pai_bench/data/prelim_01/fgis/brightness")
    base_path_2 = Path("/data1/joo/pai_bench/data/prelim_01/fgis")
    output_dir = "/data1/joo/pai_bench/results/prelim_01/metric/format/brightness"
    output_file = os.path.join(output_dir, "fgis.csv")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 분석 대상 부위 컬럼
    feature_cols = [1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13]
    
    # 2. 각 사이즈별 데이터 로드
    print("🔄 사이즈별 임베딩 데이터를 로드 중입니다...")
    df_small = load_embeddings_from_dir(base_path / "lighten/emb")
    df_regular = load_embeddings_from_dir(base_path_2 / "content/emb")
    df_big = load_embeddings_from_dir(base_path / "darken/emb")
    
    if df_small is None or df_regular is None or df_big is None:
        print("❌ 데이터를 로드하는 데 실패했습니다. 경로를 확인해주세요.")
        return

    # 공통으로 존재하는 이미지 ID 추출
    common_ids = sorted(list(set(df_small.index) & set(df_regular.index) & set(df_big.index)))
    print(f"🚀 총 {len(common_ids)}개의 공통 이미지에 대해 유사도를 계산합니다.")

    results = []

    # 3. 루프 및 계산 (S-R, R-R, R-B)
    for img_id in tqdm(common_ids, desc="Calculating FGIS similarities"):
        row_s = df_small.loc[img_id]
        row_r = df_regular.loc[img_id]
        row_b = df_big.loc[img_id]
        
        sim_sr = calculate_pair_similarity(row_s, row_r, feature_cols)
        sim_rr = calculate_pair_similarity(row_r, row_r, feature_cols)
        sim_rb = calculate_pair_similarity(row_r, row_b, feature_cols)
        
        results.append({
            'image_id': img_id,
            'fgis_lighten_orig': sim_sr,
            'fgis_orig_orig': sim_rr,
            'fgis_orig_darken': sim_rb
        })

    # 4. 결과 정리 및 평균 산출
    result_df = pd.DataFrame(results)
    
    # 평균 스코어 계산
    final_scores = result_df[['fgis_lighten_orig', 'fgis_orig_orig', 'fgis_orig_darken']].mean()
    
    print("\n" + "="*45)
    print("📊 Final Average FGIS Scores (11 Features)")
    print("-" * 45)
    print(final_scores.to_string())
    print("="*45)

    # 상세 결과 저장
    result_df.to_csv(output_file, index=False)
    print(f"✅ 결과가 {output_file}에 저장되었습니다.")

if __name__ == "__main__":
    main()