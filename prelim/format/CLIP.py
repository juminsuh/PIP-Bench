from pathlib import Path
import os
import torch
import open_clip
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

class CLIPScorer:
    def __init__(self, model_name: str, pretrained: str, device: str):
        self.device = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        print(f"🤖 Device: {self.device}")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()

    @torch.inference_mode()
    def get_features(self, img_path: str) -> torch.Tensor:
        img = Image.open(img_path).convert('RGB')
        x = self.preprocess(img).unsqueeze(0).to(self.device)
        f = self.model.encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)
        return f

def main():
    # 1. 설정
    base_path = Path("/data1/joo/pai_bench/data/prelim_01")
    output_csv = "/data1/joo/pai_bench/results/prelim_01/metric/format/brightness/clip.csv"
    
    folders = {
        "lighten": base_path / "cropped_ligthen",
        "orig": base_path / "cropped",
        "darken": base_path / "cropped_darken"
    }
    
    scorer = CLIPScorer("ViT-B-32", "openai", "cuda")
    
    # 이미지 파일 리스트 확보 (regular 폴더 기준)
    image_names = [f.name for f in folders["orig"].iterdir() if f.is_file()]
    
    all_results = []

    # 2. 루프 및 계산
    for name in tqdm(image_names, desc="Processing images"):
        paths = {k: v / name for k, v in folders.items()}
        
        # 파일이 모든 폴더에 존재하는지 확인
        if not all(p.exists() for p in paths.values()):
            continue
            
        try:
            # 특성 추출 (Features)
            feat_l = scorer.get_features(str(paths["lighten"]))
            feat_o = scorer.get_features(str(paths["orig"]))
            feat_d = scorer.get_features(str(paths["darken"]))
            
            # 유사도 계산 (Cosine Similarity)
            # 1. Small - Regular
            score_lo = torch.cosine_similarity(feat_l, feat_o).item()
            # 2. Regular - Regular (자기 자신과의 유사도 확인 또는 중복 체크용)
            score_oo = torch.cosine_similarity(feat_o, feat_o).item()
            # 3. Regular - Big
            score_od = torch.cosine_similarity(feat_o, feat_d).item()
            
            all_results.append({
                "filename": name,
                "lighten-orig": score_lo,
                "orig-orig": score_oo,
                "orig-darken": score_od
            })
            
        except Exception as e:
            print(f"Error processing {name}: {e}")

    # 3. 결과 정리 및 평균 계산
    df = pd.DataFrame(all_results)
    
    # 전체 평균 계산
    final_scores = df[["lighten-orig", "orig-orig", "orig-darken"]].mean()
    
    print("\n" + "="*30)
    print("Final Average CLIP Scores")
    print("-" * 30)
    print(final_scores)
    print("="*30)

    # 개별 결과와 평균값을 모두 저장
    df.to_csv(output_csv, index=False)
    print(f"Detailed results saved to {output_csv}")

if __name__ == "__main__":
    main()