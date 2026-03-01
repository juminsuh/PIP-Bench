import os
import shutil
import pandas as pd
from pathlib import Path

# 경로 설정
csv_dir = "/home/joo/minsuh/pai_bench/ids_grouped"
img_dir = "/data1/joo/pai_bench/img_align_celeba"
output_dir = "/data1/joo/pai_bench/celeba"

# output_dir 생성
os.makedirs(output_dir, exist_ok=True)

csv_files = sorted(Path(csv_dir).glob("*.csv"))
print(f"총 CSV 파일 수: {len(csv_files)}")

for csv_file in csv_files:
    df = pd.read_csv(csv_file, header=0)  # 첫 번째 행이 헤더 (number, id)
    
    # CSV 내 id 값 확인 (모든 행의 id가 동일하다고 가정)
    unique_ids = df['id'].unique()
    
    for person_id in unique_ids:
        id_df = df[df['id'] == person_id]
        
        # id 기반 폴더 생성 (예: /data1/joo/pai_bench/celeba/9040/)
        id_folder = os.path.join(output_dir, str(person_id))
        os.makedirs(id_folder, exist_ok=True)
        
        # 이미지 복사
        for _, row in id_df.iterrows():
            img_name = row['number']  # 예: 000007.jpg
            src = os.path.join(img_dir, img_name)
            dst = os.path.join(id_folder, img_name)
            
            if os.path.exists(src):
                shutil.copy2(src, dst)
            else:
                print(f"[경고] 이미지 없음: {src}")

print("완료!")