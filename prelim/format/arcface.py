from pathlib import Path
import os
import torch
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from insightface.app import FaceAnalysis

class ArcFaceScorer:
    def __init__(self, device: str = "cuda"):
        # providers 설정 (CUDA 사용 가능 여부에 따라 자동 선택)
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        
        self.app = FaceAnalysis(name='buffalo_l', providers=providers)
        # 이미 크롭된 얼굴들이므로 det_size는 작게, 검출 임계값은 유연하게 설정
        self.app.prepare(ctx_id=0 if device == 'cuda' else -1, det_size=(320, 320))

    def get_embedding(self, img_path: str) -> np.ndarray:
        img = cv2.imread(img_path)
        if img is None:
            return None
        
        faces = self.app.get(img)
        if not faces:
            return None
        
        # 가장 큰 얼굴(보통 크롭된 이미지이므로 첫 번째 얼굴)의 임베딩 반환
        return faces[0].normed_embedding

def main():
    # 1. 경로 설정
    base_path = Path("/data1/joo/pai_bench/data/prelim_01")
    output_csv = "/data1/joo/pai_bench/results/prelim_01/metric/format/brightness/arcface.csv"
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    folders = {
        "lighten": base_path / "cropped_ligthen",
        "orig": base_path / "cropped",
        "darken": base_path / "cropped_darken"
    }
    
    # 모델 로드 (한 번만 수행)
    scorer = ArcFaceScorer(device="cuda")
    
    # 기준 폴더(regular)에서 이미지 리스트 추출
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_names = [f.name for f in folders["orig"].iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    all_results = []

    # 2. 루프 및 계산
    for name in tqdm(image_names, desc="Calculating ArcFace scores"):
        paths = {k: v / name for k, v in folders.items()}
        
        # 모든 사이즈 폴더에 해당 파일이 있는지 확인
        if not all(p.exists() for p in paths.values()):
            continue
            
        try:
            # 임베딩 추출
            emb_s = scorer.get_embedding(str(paths["lighten"]))
            emb_r = scorer.get_embedding(str(paths["orig"]))
            emb_b = scorer.get_embedding(str(paths["darken"]))
            
            # 얼굴 검출 실패 시 skip
            if emb_s is None or emb_r is None or emb_b is None:
                continue
            
            # 코사인 유사도 계산 (ArcFace 임베딩은 이미 정규화되어 있어 내적만으로 가능)
            score_sr = float(np.dot(emb_s, emb_r))
            score_rr = float(np.dot(emb_r, emb_r))
            score_rb = float(np.dot(emb_r, emb_b))
            
            all_results.append({
                "filename": name,
                "arcface_ligthen_orig": score_sr,
                "arcface_orig_orig": score_rr,
                "arcface_orig_darken": score_rb
            })
            
        except Exception as e:
            print(f"Error processing {name}: {e}")

    # 3. 데이터프레임 생성 및 결과 요약
    if not all_results:
        print("❌ No faces were detected or processed.")
        return

    df = pd.DataFrame(all_results)
    
    # 평균 계산
    summary = df[["arcface_ligthen_orig", "arcface_orig_orig", "arcface_orig_darken"]].mean()
    
    print("\n" + "="*45)
    print(f"👤 Final Average ArcFace Scores ({len(df)} images)")
    print("-" * 45)
    print(summary.to_string())
    print("="*45)

    # 결과 저장
    df.to_csv(output_csv, index=False)
    print(f"✅ Results saved to {output_csv}")

if __name__ == "__main__":
    main()