# import os
# import base64
# import json
# import csv
# from openai import OpenAI
# from dotenv import load_dotenv
# from pathlib import Path

# # --- API 설정 (기존 유지) ---
# env_path = Path(__file__).resolve().parent.parent.parent / '.env'
# load_dotenv(dotenv_path=env_path)
# API_KEY = os.getenv("OPENAI_API_KEY")
# client = OpenAI(api_key=API_KEY)

# # --- PROMPT (기존 Continuous Score 버전 유지) ---
# SYSTEM_PROMPT = """
# You are a powerful visual expert capable of accurately analyzing faces in images and determining whether two people are the same person, based on both coarse and fine-grained facial features.
# """

# USER_PROMPT = """
# [Instruction]
# You are given two images: a reference image and a generated image.
# Your task is to evaluate whether the two images show the same person based ONLY
# on identity-related facial features. Evaluate the Rubrics carefully and follow
# the Actions exactly. Do not output anything other than the rated score.

# [Rubrics]
# 1. Determine whether the two images depict the same person based on:
#    • eyes, nose, lips, face shape, skin tone
# 2. Ignore differences from:
#    lighting, color, posture, angle, expression, hairstyle, makeup,
#    accessories, image quality.
# 3. Do not overestimate or underestimate the score. Assign the score objectively. 

# [Actions]
# 1. Compare identity-related features.
# 2. Rate how similar the identities in the two images are, from 0 (different identities) to 1 (same identities).
# """

# # --- Utils ---
# MAX_RETRIES = 3
# VALID_EXT = [".jpg", ".jpeg", ".png"]

# def encode_image(path):
#     with open(path, "rb") as f:
#         return base64.b64encode(f.read()).decode("utf-8")

# def is_valid_score(text):
#     """응답이 0~1 사이의 유효한 숫자인지 체크"""
#     try:
#         score = float(text.strip())
#         return 0 <= score <= 1
#     except ValueError:
#         return False
    
# def get_mime(path):
#     ext = os.path.splitext(path)[1].lower()
#     return "image/png" if ext == ".png" else "image/jpeg"

# # --- 핵심 실행 함수 (기존 구조 유지) ---
# def run_similarity_score(img1_path, img2_path):
#     for attempt in range(1, MAX_RETRIES + 1):
#         try:
#             img1_b64 = encode_image(img1_path)
#             img2_b64 = encode_image(img2_path)
#             img1_mime = get_mime(img1_path)
#             img2_mime = get_mime(img2_path)

#             response = client.responses.create(
#                 model="gpt-5", # 기존 모델명 유지
#                 input=[
#                     {"role": "system", "content": SYSTEM_PROMPT},
#                     {
#                         "role": "user",
#                         "content": [
#                             {"type": "input_text", "text": USER_PROMPT},
#                             {"type": "input_image", "image_url": f"data:{img1_mime};base64,{img1_b64}"},
#                             {"type": "input_image", "image_url": f"data:{img2_mime};base64,{img2_b64}"},
#                         ]
#                     },
#                 ],
#             )
            
#             result = response.output_text.strip()
#             if is_valid_score(result):
#                 return result
#             else:
#                 print(f"      [RETRY {attempt}] Invalid score format: {result}")
#         except Exception as e:
#             print(f"      [RETRY {attempt}] Error: {e}")
            
#     return "ERROR"

# # --- 메인 실행 로직 (Size 비교 및 CSV 저장) ---
# def main():
#     # 경로 설정
#     base_data_path = "/data1/joo/pai_bench/data/prelim_01"
#     dir_small = os.path.join(base_data_path, "cropped_small")
#     dir_regular = os.path.join(base_data_path, "cropped_regular")
#     dir_big = os.path.join(base_data_path, "cropped_big")
    
#     output_dir = "/data1/joo/pai_bench/result/prelim_01/metric/format/size"
#     os.makedirs(output_dir, exist_ok=True)
#     output_file = os.path.join(output_dir, "mcq_type1.csv")

#     # Regular 폴더 기준 파일 리스트업
#     files = sorted([f for f in os.listdir(dir_regular) if os.path.splitext(f)[1].lower() in VALID_EXT])
    
#     results = []

#     for fname in files:
#         print(f"🚀 Processing Similarity Size Test: {fname}")
        
#         path_small = os.path.join(dir_small, fname)
#         path_regular = os.path.join(dir_regular, fname)
#         path_big = os.path.join(dir_big, fname)

#         # 1. small_regular (Small vs Regular)
#         print(f"   [1/3] Small vs Regular...")
#         score_small_reg = run_similarity_score(path_small, path_regular)

#         # 2. regular_regular (Regular vs Regular)
#         print(f"   [2/3] Regular vs Regular...")
#         score_reg_reg = run_similarity_score(path_regular, path_regular)

#         # 3. regular_big (Regular vs Big)
#         print(f"   [3/3] Regular vs Big...")
#         score_reg_big = run_similarity_score(path_regular, path_big)

#         results.append({
#             "filename": fname,
#             "small_regular": score_small_reg,
#             "regular_regular": score_reg_reg,
#             "regular_big": score_reg_big
#         })

#     # CSV 저장 (요청하신 컬럼 형식)
#     fieldnames = ["filename", "small_regular", "regular_regular", "regular_big"]
#     with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
#         for row in results:
#             writer.writerow(row)

#     print(f"✅ Similarity Size 분석 완료! 저장 위치: {output_file}")

# if __name__ == "__main__":
#     main()

import os
import base64
import json
import csv
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

# --- API 설정 ---
env_path = Path(__file__).resolve().parent.parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- 설정 및 상수 ---
MAX_WORKERS = 15  # 계정의 RPM(분당 요청수) 제한에 따라 10~20 사이 권장
MAX_RETRIES = 3
VALID_EXT = [".jpg", ".jpeg", ".png"]

SYSTEM_PROMPT = """
You are a powerful visual expert capable of accurately analyzing faces in images and determining whether two people are the same person, based on both coarse and fine-grained facial features.
"""

USER_PROMPT = """
[Instruction]
You are given two images: a reference image and a generated image.
Your task is to evaluate whether the two images show the same person based ONLY
on identity-related facial features. Evaluate the Rubrics carefully and follow
the Actions exactly. Do not output anything other than the rated score.

[Rubrics]
1. Determine whether the two images depict the same person based on:
   • eyes, nose, lips, face shape, skin tone
2. Ignore differences from:
   lighting, color, posture, angle, expression, hairstyle, makeup,
   accessories, image quality.
3. Do not overestimate or underestimate the score. Assign the score objectively. 

[Actions]
1. Compare identity-related features.
2. Rate how similar the identities in the two images are, from 0 (different identities) to 1 (same identities).
"""

# --- Utils ---
def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_mime(path):
    ext = os.path.splitext(path)[1].lower()
    return "image/png" if ext == ".png" else "image/jpeg"

def is_valid_score(text):
    try:
        score = float(text.strip())
        return 0 <= score <= 1
    except ValueError:
        return False

# --- 핵심 실행 함수 ---
def run_similarity_score(img1_path, img2_path):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.responses.create(
                model="gpt-5",
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": USER_PROMPT},
                            {"type": "input_image", "image_url": f"data:{get_mime(img1_path)};base64,{encode_image(img1_path)}"},
                            {"type": "input_image", "image_url": f"data:{get_mime(img2_path)};base64,{encode_image(img2_path)}"},
                        ]
                    },
                ],
            )
            result = response.output_text.strip()
            if is_valid_score(result):
                return result
            time.sleep(0.5 * attempt)
        except Exception:
            time.sleep(1 * attempt)
    return "ERROR"

# --- Worker Function (Thread 단위) ---
def process_similarity_task(fname, dir_small, dir_regular, dir_big):
    p_small = os.path.join(dir_small, fname)
    p_reg = os.path.join(dir_regular, fname)
    p_big = os.path.join(dir_big, fname)

    # 3가지 케이스 실행
    return {
        "filename": fname,
        "small_regular": run_similarity_score(p_small, p_reg),
        "regular_regular": run_similarity_score(p_reg, p_reg),
        "regular_big": run_similarity_score(p_reg, p_big)
    }

# --- Main ---
def main():
    base_data_path = "/data1/joo/pai_bench/data/prelim_01"
    dir_small = os.path.join(base_data_path, "cropped_small")
    dir_regular = os.path.join(base_data_path, "cropped_regular")
    dir_big = os.path.join(base_data_path, "cropped_big")
    
    output_dir = "/data1/joo/pai_bench/result/prelim_01/metric/format/size"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "mcq_type1.csv")

    files = sorted([f for f in os.listdir(dir_regular) if os.path.splitext(f)[1].lower() in VALID_EXT])
    fieldnames = ["filename", "small_regular", "regular_regular", "regular_big"]

    print(f"🚀 Similarity Size 분석 시작: 총 {len(files)}개 파일 (Thread: {MAX_WORKERS})")

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 작업 등록
            futures = {
                executor.submit(process_similarity_task, f, dir_small, dir_regular, dir_big): f 
                for f in files
            }

            count = 0
            for future in as_completed(futures):
                fname = futures[future]
                try:
                    result = future.result()
                    writer.writerow(result)
                    csvfile.flush() # 실시간 데이터 파일 기록
                    
                    count += 1
                    if count % 10 == 0:
                        print(f"✅ 진행: {count}/{len(files)} ({(count/len(files))*100:.1f}%)")
                except Exception as e:
                    print(f"❌ {fname} 처리 중 심각한 오류: {e}")

    print(f"✨ 분석이 모두 끝났습니다! 저장 위치: {output_file}")

if __name__ == "__main__":
    main()