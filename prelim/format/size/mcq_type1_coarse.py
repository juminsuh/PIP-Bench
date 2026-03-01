# import os
# import base64
# import json
# import csv
# from openai import OpenAI
# from dotenv import load_dotenv
# from pathlib import Path
# import random

# # --- API 설정 (기존 유지) ---
# env_path = Path(__file__).resolve().parent.parent.parent / '.env'
# load_dotenv(dotenv_path=env_path)
# API_KEY = os.getenv("OPENAI_API_KEY")
# client = OpenAI(api_key=API_KEY)

# # --- PROMPT (기존 Coarse-grained 버전 유지) ---
# SYSTEM_PROMPT = """
# You are a powerful visual expert capable of accurately analyzing faces in images and determining whether two people are the same person, based on perceptual and coarse-grained impression.
# """

# USER_PROMPT = """
# [Instruction]
# You are given two images: a reference image and a generated image.
# Your task is to evaluate whether the two images show the same person based ONLY perceptual and coarse-grained impression.
# Evaluate the Rubrics carefully and follow the Actions exactly. 
# Do not output anything other than the option number you select.

# [Rubrics]
# 1. Determine whether the two images depict the same person based on perceptual impression.
# 2. Ignore fine-grained facial feature details and focus ONLY on the perceptual impression and overall identity. 
# 3. Ignore differences from:
#    lighting, color, posture, angle, expression, hairstyle, makeup,
#    accessories, image quality.
# 4.  Do not overestimate or underestimate the similarity between the two images. Choose the option objectively. 

# [Actions]
# 1. Compare perceptual impressions of two images.
# 2. Select the option that best describes the identity match between the two images. 
# 3. Return ONLY the single number (e.g., 1).

# [Options]
# {options}
# """

# OPTIONS = [
#     "They are totally different people.",
#     "They are quite different people.",
#     "It is unclear whether they are the same person or not.",
#     "They look similar, but are not same people.",
#     "They are same people, even they are might under slight different conditions (e.g., lighting, angle, style).",
# ]

# TEXT_TO_SCORE = {
#     "They are totally different people.": 0.0,
#     "They are quite different people.": 0.25,
#     "It is unclear whether they are the same person or not.": None,
#     "They look similar, but are not same people.": 0.75,
#     "They are same people, even they are might under slight different conditions (e.g., lighting, angle, style).": 1.0,
# }

# # --- Utils ---
# MAX_RETRIES = 3
# VALID_EXT = [".jpg", ".jpeg", ".png"]

# def is_valid(text, num_to_text):
#     return text.strip() in num_to_text
    
# def build_shuffled_prompt():
#     shuffled = OPTIONS.copy()
#     random.shuffle(shuffled)
#     options_text = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(shuffled))
#     num_to_text = {str(i+1): opt for i, opt in enumerate(shuffled)}
#     return USER_PROMPT.format(options=options_text), num_to_text

# def encode_image(path):
#     with open(path, "rb") as f:
#         return base64.b64encode(f.read()).decode("utf-8")

# def get_mime(path):
#     ext = os.path.splitext(path)[1].lower()
#     return "image/png" if ext == ".png" else "image/jpeg"

# # --- 핵심 실행 함수 (Coarse 전용) ---
# def run_type1_coarse_score(img1_path, img2_path):
#     for attempt in range(1, MAX_RETRIES + 1):
#         try:
#             user_prompt, num_to_text = build_shuffled_prompt()
#             img1_b64 = encode_image(img1_path)
#             img2_b64 = encode_image(img2_path)
#             img1_mime = get_mime(img1_path)
#             img2_mime = get_mime(img2_path)

#             response = client.responses.create(
#                 model="gpt-5",
#                 input=[
#                     {"role": "system", "content": SYSTEM_PROMPT},
#                     {
#                         "role": "user",
#                         "content": [
#                             {"type": "input_text", "text": user_prompt},
#                             {"type": "input_image", "image_url": f"data:{img1_mime};base64,{img1_b64}"},
#                             {"type": "input_image", "image_url": f"data:{img2_mime};base64,{img2_b64}"},
#                         ]
#                     },
#                 ],
#             )
            
#             result = response.output_text.strip()
#             if is_valid(result, num_to_text):
#                 chosen_text = num_to_text[result]
#                 return TEXT_TO_SCORE[chosen_text]
            
#             print(f"      [RETRY {attempt}] Invalid response: {result}")
#         except Exception as e:
#             print(f"      [RETRY {attempt}] Error: {e}")
            
#     return -1.0

# # --- 메인 실행 로직 (Size 비교 및 CSV 저장) ---
# def main():
#     # 경로 설정
#     base_data_path = "/data1/joo/pai_bench/data/prelim_01"
#     dir_small = os.path.join(base_data_path, "cropped_small")
#     dir_regular = os.path.join(base_data_path, "cropped_regular")
#     dir_big = os.path.join(base_data_path, "cropped_big")
    
#     output_dir = "/data1/joo/pai_bench/result/prelim_01/metric/format/size"
#     os.makedirs(output_dir, exist_ok=True)
#     output_file = os.path.join(output_dir, "mcq_type1_coarse.csv")

#     # Regular 폴더 기준 파일 리스트업
#     files = sorted([f for f in os.listdir(dir_regular) if os.path.splitext(f)[1].lower() in VALID_EXT])
    
#     results = []

#     for fname in files:
#         print(f"🚀 Processing Coarse Size Test: {fname}")
        
#         path_small = os.path.join(dir_small, fname)
#         path_regular = os.path.join(dir_regular, fname)
#         path_big = os.path.join(dir_big, fname)

#         # 3가지 케이스 비교 수행
#         print(f"   [1/3] Small vs Regular...")
#         score_small_reg = run_type1_coarse_score(path_small, path_regular)

#         print(f"   [2/3] Regular vs Regular...")
#         score_reg_reg = run_type1_coarse_score(path_regular, path_regular)

#         print(f"   [3/3] Regular vs Big...")
#         score_reg_big = run_type1_coarse_score(path_regular, path_big)

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

#     print(f"✅ Coarse Size 분석 완료! 저장 위치: {output_file}")

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
MAX_WORKERS = 15  # 네트워크 상태와 API 티어에 따라 10~20 사이 조절 권장
MAX_RETRIES = 3
VALID_EXT = [".jpg", ".jpeg", ".png"]

SYSTEM_PROMPT = """
You are a powerful visual expert capable of accurately analyzing faces in images and determining whether two people are the same person, based on perceptual and coarse-grained impression.
"""

USER_PROMPT = """
[Instruction]
You are given two images: a reference image and a generated image.
Your task is to evaluate whether the two images show the same person based ONLY perceptual and coarse-grained impression.
Evaluate the Rubrics carefully and follow the Actions exactly. 
Do not output anything other than the option number you select.

[Rubrics]
1. Determine whether the two images depict the same person based on perceptual impression.
2. Ignore fine-grained facial feature details and focus ONLY on the perceptual impression and overall identity. 
3. Ignore differences from:
   lighting, color, posture, angle, expression, hairstyle, makeup,
   accessories, image quality.
4. Do not overestimate or underestimate the similarity between the two images. Choose the option objectively. 

[Actions]
1. Compare perceptual impressions of two images.
2. Select the option that best describes the identity match between the two images. 
3. Return ONLY the single number (e.g., 1).

[Options]
{options}
"""

OPTIONS = [
    "They are totally different people.",
    "They are quite different people.",
    "It is unclear whether they are the same person or not.",
    "They look similar, but are not same people.",
    "They are same people, even they are might under slight different conditions (e.g., lighting, angle, style).",
]

TEXT_TO_SCORE = {
    "They are totally different people.": 0.0,
    "They are quite different people.": 0.25,
    "It is unclear whether they are the same person or not.": None,
    "They look similar, but are not same people.": 0.75,
    "They are same people, even they are might under slight different conditions (e.g., lighting, angle, style).": 1.0,
}

# --- Utils ---
def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_mime(path):
    ext = os.path.splitext(path)[1].lower()
    return "image/png" if ext == ".png" else "image/jpeg"

def build_shuffled_prompt():
    shuffled = OPTIONS.copy()
    random.shuffle(shuffled)
    options_text = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(shuffled))
    num_to_text = {str(i+1): opt for i, opt in enumerate(shuffled)}
    return USER_PROMPT.format(options=options_text), num_to_text

# --- API 호출 함수 ---
def run_type1_coarse_score(img1_path, img2_path):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            user_prompt, num_to_text = build_shuffled_prompt()
            response = client.responses.create(
                model="gpt-5",
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": user_prompt},
                            {"type": "input_image", "image_url": f"data:{get_mime(img1_path)};base64,{encode_image(img1_path)}"},
                            {"type": "input_image", "image_url": f"data:{get_mime(img2_path)};base64,{encode_image(img2_path)}"},
                        ]
                    },
                ],
            )
            result = response.output_text.strip()
            if result in num_to_text:
                chosen_text = num_to_text[result]
                return TEXT_TO_SCORE[chosen_text]
            
            time.sleep(0.5 * attempt)
        except Exception:
            time.sleep(1 * attempt)
    return -1.0

# --- Worker Function (Thread 단위 작업) ---
def process_size_task(fname, dir_small, dir_regular, dir_big):
    """파일 하나에 대해 3가지 케이스를 모두 실행하고 결과를 딕셔너리로 반환"""
    path_small = os.path.join(dir_small, fname)
    path_regular = os.path.join(dir_regular, fname)
    path_big = os.path.join(dir_big, fname)

    return {
        "filename": fname,
        "small_regular": run_type1_coarse_score(path_small, path_regular),
        "regular_regular": run_type1_coarse_score(path_regular, path_regular),
        "regular_big": run_type1_coarse_score(path_regular, path_big)
    }

# --- Main ---
def main():
    # 경로 설정
    base_data_path = "/data1/joo/pai_bench/data/prelim_01"
    dir_small = os.path.join(base_data_path, "cropped_small")
    dir_regular = os.path.join(base_data_path, "cropped_regular")
    dir_big = os.path.join(base_data_path, "cropped_big")
    
    output_dir = "/data1/joo/pai_bench/result/prelim_01/metric/format/size"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "mcq_type1_coarse.csv")

    files = sorted([f for f in os.listdir(dir_regular) if os.path.splitext(f)[1].lower() in VALID_EXT])
    fieldnames = ["filename", "small_regular", "regular_regular", "regular_big"]

    print(f"🚀 Coarse Size 멀티스레딩 분석 시작: 총 {len(files)}개 파일 (Workers: {MAX_WORKERS})")

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # ThreadPoolExecutor를 사용한 병렬 처리
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 작업 등록 (Future 객체 생성)
            future_to_file = {
                executor.submit(process_size_task, f, dir_small, dir_regular, dir_big): f 
                for f in files
            }

            completed_count = 0
            for future in as_completed(future_to_file):
                fname = future_to_file[future]
                try:
                    result = future.result()
                    writer.writerow(result)
                    csvfile.flush() # 각 작업 완료 시 파일에 물리적으로 기록 (유실 방지)
                    
                    completed_count += 1
                    if completed_count % 10 == 0:
                        print(f"✅ 진행도: {completed_count}/{len(files)} ({(completed_count/len(files))*100:.1f}%)")
                except Exception as e:
                    print(f"❌ {fname} 처리 중 에러 발생: {e}")

    print(f"✨ 모든 작업 완료! 결과 파일: {output_file}")

if __name__ == "__main__":
    main()