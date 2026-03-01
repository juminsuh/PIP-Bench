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

# # --- PROMPT (Fine-grained 버전 유지) ---
# SYSTEM_PROMPT = """
# You are a powerful visual expert capable of accurately analyzing faces in images and determining whether two people share same fine-grained facial features.
# """

# USER_PROMPT = """
# [Instruction]
# You are given two images: a reference image and a generated image.
# Your task is to evaluate whether the two images share same identity-related facial features. Evaluate the Rubrics carefully and follow
# the Actions exactly. Do not output anything other than the option number(s) you select.

# [Rubrics]
# 1. Determine whether the two images depict the same facial features based on:
#    • eyes, nose, lips, face shape, skin tone, spatial arrangement of their features
# 2. Ignore perceptual and coarse-grained impression and focus ONLY on the detailed facial features. 
# 3. Ignore differences from:
#    lighting, color, posture, angle, expression, hairstyle, makeup,
#    accessories, image quality.
# 4.  Do not overestimate or underestimate the decision. Choose the option objectively. 

# [Actions]
# 1. Compare identity-related facial features.
# 2. If the entire facial features are same → select the option starts with “Yes”. 
# 3. If any identity feature differs → select one or more options that start with "No".
# 4. You may select multiple options, but you must not choose an option starting with "Yes" together with any option starting with "No."
# 5. Return only the option number(s):
# • If only one option is selected → return a single number (e.g., 1).
# • If more than one options are selected  → return all option numbers separated by commas (e.g., 2, 4, 5)

# [Options]
# {options}
# """

# OPTIONS = [
#     "Yes, their all facial features including eyes, noses, lips, face shapes, skin tones, and spatial arrangement of their features are preserved.",
#     "No, their eyes are quite different.",
#     "No, their noses are quite different.",
#     "No, their lips are quite different.",
#     "No, their face shapes are quite different.",
#     "No, their skin tones are quite different.",
#     "No, the spatial arrangement of their features are quite different."
# ]

# # --- Utils (기존 Fine-grained 로직 유지) ---
# MAX_RETRIES = 3
# VALID_EXT = [".jpg", ".jpeg", ".png"]

# def is_valid(text, num_to_text):
#     parts = [p.strip() for p in text.strip().split(",")]
#     return all(p in num_to_text for p in parts)

# def parse_response(text, num_to_text):
#     parts = [p.strip() for p in text.strip().split(",")]
#     chosen_texts = [num_to_text[p] for p in parts]

#     if any(t.startswith("Yes") for t in chosen_texts):
#         if len(chosen_texts) == 1:
#             return 1.0
#         else:
#             return None # Invalid case (Yes + No)
#     else:
#         num_mistakes = len(chosen_texts)
#         score = max(0.0, 1.0 - (1.0 / 6.0) * num_mistakes)
#         return round(score, 4)

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

# # --- 핵심 실행 함수 (Fine-grained 전용) ---
# def run_type1_fine_score(img1_path, img2_path):
#     for attempt in range(1, MAX_RETRIES + 1):
#         try:
#             user_prompt, num_to_text = build_shuffled_prompt()
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
#                             {"type": "input_text", "text": user_prompt},
#                             {"type": "input_image", "image_url": f"data:{img1_mime};base64,{img1_b64}"},
#                             {"type": "input_image", "image_url": f"data:{img2_mime};base64,{img2_b64}"},
#                         ]
#                     },
#                 ],
#             )
            
#             result = response.output_text.strip()
#             if is_valid(result, num_to_text):
#                 score = parse_response(result, num_to_text)
#                 if score is not None:
#                     return score
            
#             print(f"      [RETRY {attempt}] Invalid or inconsistent response: {result}")
#         except Exception as e:
#             print(f"      [RETRY {attempt}] Error: {e}")
            
#     return -1.0

# # --- 메인 실행 로직 (Brightness 비교 및 CSV 저장) ---
# def main():
#     # 경로 설정
#     base_data_path = "/data1/joo/pai_bench/data/prelim_01"
#     dir_orig = os.path.join(base_data_path, "cropped")
#     dir_lighten = os.path.join(base_data_path, "cropped_lighten")
#     dir_darken = os.path.join(base_data_path, "cropped_darken")
    
#     output_dir = "/data1/joo/pai_bench/result/prelim_01/metric/format/brightness"
#     os.makedirs(output_dir, exist_ok=True)
#     output_file = os.path.join(output_dir, "mcq_type1_fine.csv") # 파일명 구분

#     # 원본 폴더 기준 파일 리스트업
#     files = sorted([f for f in os.listdir(dir_orig) if os.path.splitext(f)[1].lower() in VALID_EXT])
    
#     results = []

#     for fname in files:
#         print(f"🚀 Processing Fine-grained: {fname}")
        
#         path_orig = os.path.join(dir_orig, fname)
#         path_lighten = os.path.join(dir_lighten, fname)
#         path_darken = os.path.join(dir_darken, fname)

#         # 1. mcq_type1_lighten_orig
#         print(f"   [1/3] Lighten vs Orig...")
#         score_lighten = run_type1_fine_score(path_lighten, path_orig)

#         # 2. mcq_type1_orig_orig
#         print(f"   [2/3] Orig vs Orig...")
#         score_orig = run_type1_fine_score(path_orig, path_orig)

#         # 3. mcq_type1_orig_darken
#         print(f"   [3/3] Orig vs Darken...")
#         score_darken = run_type1_fine_score(path_orig, path_darken)

#         results.append({
#             "filename": fname,
#             "mcq_type1_lighten_orig": score_lighten,
#             "mcq_type1_orig_orig": score_orig,
#             "mcq_type1_orig_darken": score_darken
#         })

#     # CSV 저장 (요청하신 컬럼 형식)
#     fieldnames = ["filename", "mcq_type1_lighten_orig", "mcq_type1_orig_orig", "mcq_type1_orig_darken"]
#     with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
#         for row in results:
#             writer.writerow(row)

#     print(f"✅ Fine-grained 분석 완료! 저장 위치: {output_file}")

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
MAX_WORKERS = 15  # 계정 Tier에 따라 10~20 사이 권장
MAX_RETRIES = 3
VALID_EXT = [".jpg", ".jpeg", ".png"]

SYSTEM_PROMPT = """
You are a powerful visual expert capable of accurately analyzing faces in images and determining whether two people share same fine-grained facial features.
"""

USER_PROMPT = """
[Instruction]
You are given two images: a reference image and a generated image.
Your task is to evaluate whether the two images share same identity-related facial features. Evaluate the Rubrics carefully and follow
the Actions exactly. Do not output anything other than the option number(s) you select.

[Rubrics]
1. Determine whether the two images depict the same facial features based on:
   • eyes, nose, lips, face shape, skin tone, spatial arrangement of their features
2. Ignore perceptual and coarse-grained impression and focus ONLY on the detailed facial features. 
3. Ignore differences from:
   lighting, color, posture, angle, expression, hairstyle, makeup,
   accessories, image quality.
4. Do not overestimate or underestimate the decision. Choose the option objectively. 

[Actions]
1. Compare identity-related facial features.
2. If the entire facial features are same → select the option starts with “Yes”. 
3. If any identity feature differs → select one or more options that start with "No".
4. You may select multiple options, but you must not choose an option starting with "Yes" together with any option starting with "No."
5. Return only the option number(s):
• If only one option is selected → return a single number (e.g., 1).
• If more than one options are selected  → return all option numbers separated by commas (e.g., 2, 4, 5)

[Options]
{options}
"""

OPTIONS = [
    "Yes, their all facial features including eyes, noses, lips, face shapes, skin tones, and spatial arrangement of their features are preserved.",
    "No, their eyes are quite different.",
    "No, their noses are quite different.",
    "No, their lips are quite different.",
    "No, their face shapes are quite different.",
    "No, their skin tones are quite different.",
    "No, the spatial arrangement of their features are quite different."
]

TEXT_TO_SCORE = {
    "Yes": 1.0, # Prefix check용
}

# --- Utils ---
def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_mime(path):
    ext = os.path.splitext(path)[1].lower()
    return "image/png" if ext == ".png" else "image/jpeg"

def is_valid(text, num_to_text):
    parts = [p.strip() for p in text.strip().split(",")]
    return all(p in num_to_text for p in parts)

def parse_response(text, num_to_text):
    parts = [p.strip() for p in text.strip().split(",")]
    chosen_texts = [num_to_text[p] for p in parts]

    if any(t.startswith("Yes") for t in chosen_texts):
        return 1.0 if len(chosen_texts) == 1 else None
    else:
        num_mistakes = len(chosen_texts)
        score = max(0.0, 1.0 - (1.0 / 6.0) * num_mistakes)
        return round(score, 4)

def build_shuffled_prompt():
    shuffled = OPTIONS.copy()
    random.shuffle(shuffled)
    options_text = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(shuffled))
    num_to_text = {str(i+1): opt for i, opt in enumerate(shuffled)}
    return USER_PROMPT.format(options=options_text), num_to_text

# --- 실행 함수 ---
def run_type1_fine_score(img1_path, img2_path):
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
            if is_valid(result, num_to_text):
                score = parse_response(result, num_to_text)
                if score is not None:
                    return score
            time.sleep(0.5 * attempt)
        except Exception:
            time.sleep(1 * attempt)
    return -1.0

# --- Worker Function ---
def process_fine_grained_task(fname, dir_orig, dir_lighten, dir_darken):
    path_orig = os.path.join(dir_orig, fname)
    path_lighten = os.path.join(dir_lighten, fname)
    path_darken = os.path.join(dir_darken, fname)

    return {
        "filename": fname,
        "mcq_type1_lighten_orig": run_type1_fine_score(path_lighten, path_orig),
        "mcq_type1_orig_orig": run_type1_fine_score(path_orig, path_orig),
        "mcq_type1_orig_darken": run_type1_fine_score(path_orig, path_darken)
    }

# --- Main ---
def main():
    base_data_path = "/data1/joo/pai_bench/data/prelim_01"
    dir_orig = os.path.join(base_data_path, "cropped")
    dir_lighten = os.path.join(base_data_path, "cropped_lighten")
    dir_darken = os.path.join(base_data_path, "cropped_darken")
    
    output_dir = "/data1/joo/pai_bench/result/prelim_01/metric/format/brightness"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "mcq_type1_fine.csv")

    files = sorted([f for f in os.listdir(dir_orig) if os.path.splitext(f)[1].lower() in VALID_EXT])
    fieldnames = ["filename", "mcq_type1_lighten_orig", "mcq_type1_orig_orig", "mcq_type1_orig_darken"]

    print(f"🚀 Fine-grained 분석 시작: 총 {len(files)}개 파일 (스레드: {MAX_WORKERS})")

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_file = {
                executor.submit(process_fine_grained_task, f, dir_orig, dir_lighten, dir_darken): f 
                for f in files
            }

            count = 0
            for future in as_completed(future_to_file):
                try:
                    result = future.result()
                    writer.writerow(result)
                    csvfile.flush() # 실시간 저장
                    
                    count += 1
                    if count % 10 == 0:
                        print(f"✅ 진행도: {count}/{len(files)} ({(count/len(files))*100:.1f}%)")
                except Exception as e:
                    fn = future_to_file[future]
                    print(f"❌ {fn} 처리 중 오류 발생: {e}")

    print(f"✨ 분석 완료! 결과가 {output_file}에 저장되었습니다.")

if __name__ == "__main__":
    main()