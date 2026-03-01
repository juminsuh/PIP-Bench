import pandas as pd
import os

def calculate_average_scores(coarse_path, fine_path, output_path):
    # 1. 파일 불러오기
    if not os.path.exists(coarse_path) or not os.path.exists(fine_path):
        print("❌ 입력 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    # [수정] dtype={'image0': str, 'image1': str} 를 추가하여 0001 형태를 유지합니다.
    df_coarse = pd.read_csv(coarse_path, dtype={'image0': str, 'image1': str})
    df_fine = pd.read_csv(fine_path, dtype={'image0': str, 'image1': str})

    # 2. image0, image1을 기준으로 두 데이터프레임 병합
    merged = pd.merge(
        df_coarse, 
        df_fine, 
        on=['image0', 'image1'], 
        suffixes=('_coarse', '_fine')
    )

    # 3. Score 평균 계산
    # 컬럼명이 파일마다 다를 수 있으니 확인이 필요합니다. 
    # 일반적인 경우에는 merged[['score_coarse', 'score_fine']].mean(axis=1) 형태일 것입니다.
    # 제공해주신 이전 코드의 컬럼명을 유지합니다.
    score_cols = ['mcq_type1_coarse_score', 'mcq_type1_fine_score']
    merged['score'] = merged[score_cols].mean(axis=1)

    # 4. 결과 컬럼 정리
    result_df = merged[['image0', 'image1', 'score']].copy()

    # 5. 결과 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False)

    print(f"✅ 처리가 완료되었습니다! (문자열 포맷 유지)")
    print(f"📍 저장 위치: {output_path}")
    print(f"📊 총 {len(result_df)}개의 데이터가 처리되었습니다.")

if __name__ == "__main__":
    base_dir = "/data1/joo/pai_bench/result/prelim_01/metric/content"
    
    coarse_csv = os.path.join(base_dir, "mcq_type1_coarse.csv")
    fine_csv = os.path.join(base_dir, "mcq_type1_fine.csv")
    output_csv = os.path.join(base_dir, "mcq_type1.csv")

    calculate_average_scores(coarse_csv, fine_csv, output_csv)