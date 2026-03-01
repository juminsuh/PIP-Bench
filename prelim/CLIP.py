"""
CLIP-I scorer for celeb face angle dataset
"""

from pathlib import Path
import re
import os
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter
from tqdm import tqdm

import torch
import open_clip

class CLIPScorer:
    def __init__(self, model_name: str, pretrained: str, device: str):
        self.device = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        print(f"🤖 device: {self.device}")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()

    @torch.inference_mode()
    def img_feat(self, img: Image.Image) -> torch.Tensor:
        x = self.preprocess(img).unsqueeze(0).to(self.device)
        f = self.model.encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)
        return f.squeeze(0)
    
    def clip_score(self, img1_path: str, img2_path: str) -> float:
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        feat1 = self.img_feat(img1)
        feat2 = self.img_feat(img2)
        
        score = torch.cosine_similarity(feat1, feat2, dim=0)
        return score.item()
    
    def compare_images_in_folder(self, reference_img_path: str, folder_path: str):
        folder = Path(folder_path)
        reference_path = Path(reference_img_path)
        
        if not reference_path.exists():
            raise FileNotFoundError(f"Reference image not found: {reference_img_path}")
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = [f for f in folder.iterdir() 
                      if f.is_file() and f.suffix.lower() in image_extensions]
        
        if not image_files:
            raise ValueError(f"No image files found in folder: {folder_path}")
        
        results = []
        
        reference_id = os.path.basename(reference_path).split(".")[0]
        
        for img_file in tqdm(image_files, desc="Calculating CLIP scores"):
            try:
                score = self.clip_score(str(reference_path), str(img_file))
                image_id = os.path.basename(img_file).split(".")[0]
                results.append({
                    'image0': reference_id,
                    'image1': image_id,
                    'clip_score': score
                })
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                continue
            
        return results



def main():
    images_folder = "/data1/joo/pai_bench/data/prelim_01/cropped_big"
    output_csv = "/data1/joo/pai_bench/results/prelim_01/metric/format/size/clip.csv"
    total_results = []

    model_name = "ViT-B-32"
    pretrained = "openai"
    device = "cuda"
    
    for i, filename in enumerate(os.listdir(images_folder)):
        print(f"➡️ Processing {i+1} image...")
        reference_image = os.path.join(images_folder, filename)
    
        scorer = CLIPScorer(model_name, pretrained, device)
        results = scorer.compare_images_in_folder(reference_image, images_folder)
        
        total_results.extend(results)

    df = pd.DataFrame(total_results)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    main()