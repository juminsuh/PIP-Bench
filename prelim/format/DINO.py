from pathlib import Path
import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

class DINOScorer:
    def __init__(self, model_name: str = "facebook/dinov2-base", device: str = "cuda"):
        self.device = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        print(f"🤖 Device: {self.device}")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def get_features(self, img_path: str) -> torch.Tensor:
        img = Image.open(img_path).convert('RGB')
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        # Global average pooling (CLS 토큰 대신 patch들의 평균 사용 가능, 여기선 CLS 토큰 사용)
        f = outputs.last_hidden_state[:, 0, :] 
        f = f / f.norm(dim=-1, keepdim=True)
        return f

def main():
    # 1. 경로 설정
    base_path = Path("/data1/joo/pai_bench/data/prelim_01")
    output_csv = "/data1/joo/pai_bench/results/prelim_01/metric/format/brightness/dino.csv"
    
    # 출력 폴더가 없으면 생성
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    folders = {
        "lighten": base_path / "cropped_ligthen",
        "orig": base_path / "cropped",
        "darken": base_path / "cropped_darken"
    }
    
    # 모델 로드 (한 번만 수행)
    scorer = DINOScorer(device="cuda")
    
    # 기준 폴더(regular)에서 이미지 파일 목록 가져오기
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_names = [f.name for f in folders["orig"].iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    all_results = []

    # 2. 루프 및 계산
    for name in tqdm(image_names, desc="Calculating DINO scores"):
        paths = {k: v / name for k, v in folders.items()}
        
        # 세 폴더 모두에 파일이 존재하는지 확인
        if not all(p.exists() for p in paths.values()):
            continue
            
        try:
            # 특성 추출 (Features)
            feat_s = scorer.get_features(str(paths["lighten"]))
            feat_r = scorer.get_features(str(paths["orig"]))
            feat_b = scorer.get_features(str(paths["darken"]))
            
            # 코사인 유사도 계산
            score_sr = torch.cosine_similarity(feat_s, feat_r).item()
            score_rr = torch.cosine_similarity(feat_r, feat_r).item()
            score_rb = torch.cosine_similarity(feat_r, feat_b).item()
            
            all_results.append({
                "filename": name,
                "dino_lighten_orig": score_sr,
                "dino_orig_orig": score_rr,
                "dino_orig_darken": score_rb
            })
            
        except Exception as e:
            print(f"Error processing {name}: {e}")

    # 3. 데이터프레임 생성 및 평균 계산
    if not all_results:
        print("No results were calculated. Check your file paths.")
        return

    df = pd.DataFrame(all_results)
    
    # 최종 평균 스코어 계산
    avg_sr = df["dino_lighten_orig"].mean()
    avg_rr = df["dino_orig_orig"].mean()
    avg_rb = df["dino_orig_darken"].mean()
    
    print("\n" + "="*40)
    print(f"📊 Final Average DINO Scores ({len(df)} images)")
    print("-" * 40)
    print(f"Small  ↔ Regular: {avg_sr:.4f}")
    print(f"Regular ↔ Regular: {avg_rr:.4f}")
    print(f"Regular ↔ Big    : {avg_rb:.4f}")
    print("="*40)

    # 전체 결과 저장
    df.to_csv(output_csv, index=False)
    print(f"✅ Results saved to {output_csv}")

if __name__ == "__main__":
    main()