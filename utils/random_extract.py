import os
import random
import shutil
from pathlib import Path

def pick_balanced_random_images(base_path, total_count=50):
    # 1. 모델 폴더 목록 가져오기
    models = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    num_models = len(models)
    
    if num_models == 0:
        print("❌ 모델 폴더를 찾을 수 없습니다.")
        return []

    # 모델당 기본 할당량 계산
    per_model_count = total_count // num_models
    extra_count = total_count % num_models
    
    selected_images = []
    all_available_remaining = [] # 할당량 초과분을 위한 예비 리스트

    valid_extensions = ('.jpg', '.jpeg', '.png')

    print(f"📊 총 {num_models}개 모델 감지. 모델당 기본 {per_model_count}장씩 추출합니다.")

    for i, model in enumerate(models):
        model_path = os.path.join(base_path, model)
        model_images = []

        # 해당 모델 하위의 모든 이미지 수집
        for root, dirs, files in os.walk(model_path):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    model_images.append(os.path.join(root, file))

        # 현재 모델에서 뽑아야 할 개수 결정 (나머지 2장은 앞쪽 모델에서 하나씩 더 가져옴)
        n_to_pick = per_model_count + (1 if i < extra_count else 0)
        
        if len(model_images) >= n_to_pick:
            picked = random.sample(model_images, n_to_pick)
            selected_images.extend(picked)
        else:
            # 모델 내 이미지 수가 부족할 경우 있는 대로 다 넣음
            selected_images.extend(model_images)
            print(f"⚠️ {model} 모델의 이미지가 부족합니다 ({len(model_images)}장 발견).")

    return selected_images

if __name__ == "__main__":
    base_path = "/data1/joo/pai_bench/data/generation"
    output_dir = "/data1/joo/pai_bench/data/compare/text"
    
    # 50장 균등 추출
    final_samples = pick_balanced_random_images(base_path, 50)
    random.shuffle(final_samples) # 순서 섞기

    # 결과 확인 및 복사
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n✅ 균등하게 선택된 {len(final_samples)}장의 이미지 목록:")
    
    for i, img_path in enumerate(final_samples):
        path_obj = Path(img_path)
        
        # 구조: /.../generation/[model_name]/[id_folder]/[filename]
        # path_obj.parts를 이용해 뒤에서 두 번째 요소(id_folder)를 가져옵니다.
        model_name = path_obj.parts[-3]  # 모델명 (e.g., gemini)
        id_folder = path_obj.parts[-2]   # ID 폴더명 (e.g., 001)
        file_name = path_obj.name        # 실제 파일명
        
        if i < 5:
            print(f"[{i+1:02d}] From Folder: {id_folder} | Model: {model_name} | File: {file_name}")
        
        # 파일명 변경: 순번_ID폴더_모델명_원본이름
        # 예: 001_id025_gemini_output.png
        dest_name = f"{i+1:03d}_id{id_folder}_{model_name}_{file_name}"
        shutil.copy(img_path, os.path.join(output_dir, dest_name))

    print(f"\n📂 모든 파일이 '{output_dir}'에 저장되었습니다.")