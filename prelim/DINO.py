"""
DINO scorer for celeb face angle dataset
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
import torchvision.transforms as transforms
from transformers import AutoImageProcessor, AutoModel

class DINOScorer:
    def __init__(self, model_name: str = "facebook/dinov2-base", device: str = "cuda"):
        self.device = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def img_feat(self, img: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        f = outputs.last_hidden_state.mean(dim=1)  # Global average pooling
        f = f / f.norm(dim=-1, keepdim=True)
        return f.squeeze(0)
    
    def dino_score(self, img1_path: str, img2_path: str) -> float:
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

        for img_file in tqdm(image_files, desc="Calculating DINO scores"):
            try:
                score = self.dino_score(str(reference_path), str(img_file))
                image_id = os.path.basename(img_file).split(".")[0]
                results.append({
                    'image0': reference_id,
                    'image1': image_id,
                    'dino_score': score
                })
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                continue
        
        return results


def main():
    images_folder = "/data1/joo/pai_bench/data/prelim_01/cropped"
    output_csv = "/data1/joo/pai_bench/results/prelim_01/dino.csv"
    total_results = []
    device = "cuda"

    for i, filename in enumerate(os.listdir(images_folder)):
        print(f"➡️ Processing {i+1} image...")
        reference_image = os.path.join(images_folder, filename)
        
        scorer = DINOScorer(device=device)
        results = scorer.compare_images_in_folder(reference_image, images_folder)
        total_results.extend(results)

    df = pd.DataFrame(total_results)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    main()