# import os
# import base64
# import json
# import csv
# from openai import OpenAI
# from dotenv import load_dotenv
# from pathlib import Path
# import random

# # --- API 설정 (기존 유지) ---
# env_path = Path(__file__).resolve().parent.parent.parent.parent / '.env'
# load_dotenv(dotenv_path=env_path)
# API_KEY = os.getenv("OPENAI_API_KEY")
# client = OpenAI(api_key=API_KEY)

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

# # --- Utils (기존 유지) ---
# MAX_RETRIES = 3
# VALID_EXT = [".jpg", ".jpeg", ".png"]

# def is_valid(text, num_to_text):
#     return text.strip() in num_to_text
    
# def build_shuffled_prompt():
#     shuffled = OPTIONS.copy()
#     random.shuffle(shuffled)
#     options_text = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(shuffled))
#     num_to_text = {str(i+1): opt for i, opt in enumerate(shuffled)}
#     prompt = USER_PROMPT.format(options=options_text)
#     return prompt, num_to_text

# def encode_image(path):
#     with open(path, "rb") as f:
#         return base64.b64encode(f.read()).decode("utf-8")

# def get_mime(path):
#     ext = os.path.splitext(path)[1].lower()
#     return "image/png" if ext == ".png" else "image/jpeg"

# # --- 핵심 실행 함수 (기존 client.responses.create 구조 유지) ---
# def run_type1_mcq(img1_path, img2_path):
#     for attempt in range(1, MAX_RETRIES + 1):
#         try:
#             user_prompt, num_to_text = build_shuffled_prompt()
#             img1_b64 = encode_image(img1_path)
#             img2_b64 = encode_image(img2_path)
#             img1_mime = get_mime(img1_path)
#             img2_mime = get_mime(img2_path)

#             # 기존 코드의 호출 방식 그대로 유지
#             response = client.responses.create(
#                 model="gpt-5", # 사용자님의 환경에 설정된 모델명 유지
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
#             else:
#                 print(f"      [RETRY {attempt}] Invalid response: {result}")
#         except Exception as e:
#             print(f"      [RETRY {attempt}] Error: {e}")
            
#     return -1.0 # 실패 시

# # --- 메인 실행 로직 (요청하신 경로 및 저장 형식 반영) ---
# def main():
#     # 경로 설정
#     base_data_path = "/data1/joo/pai_bench/data/prelim_01"
#     dir_orig = os.path.join(base_data_path, "cropped")
#     dir_lighten = os.path.join(base_data_path, "cropped_lighten")
#     dir_darken = os.path.join(base_data_path, "cropped_darken")

#     output_dir = "/data1/joo/pai_bench/result/prelim_01/metric/format/brightness"
#     os.makedirs(output_dir, exist_ok=True)
#     output_file = os.path.join(output_dir, "mcq_type1_coarse.csv")

#     # 원본 폴더를 기준으로 파일 목록 생성
#     files = sorted([f for f in os.listdir(dir_orig) if os.path.splitext(f)[1].lower() in VALID_EXT])
    
#     results = []

#     for fname in files:
#         print(f"🚀 Processing: {fname}")
        
#         path_orig = os.path.join(dir_orig, fname)
#         path_lighten = os.path.join(dir_lighten, fname)
#         path_darken = os.path.join(dir_darken, fname)

#         # 3가지 케이스 비교 수행
#         # 1. mcq_type1_lighten_orig
#         print(f"   [1/3] Lighten vs Orig...")
#         score_lighten = run_type1_mcq(path_lighten, path_orig)

#         # 2. mcq_type1_orig_orig
#         print(f"   [2/3] Orig vs Orig...")
#         score_orig = run_type1_mcq(path_orig, path_orig)

#         # 3. mcq_type1_orig_darken
#         print(f"   [3/3] Orig vs Darken...")
#         score_darken = run_type1_mcq(path_orig, path_darken)

#         results.append({
#             "filename": fname,
#             "mcq_type1_lighten_orig": score_lighten,
#             "mcq_type1_orig_orig": score_orig,
#             "mcq_type1_orig_darken": score_darken
#         })

#     # CSV 저장 (요청하신 컬럼 순서)
#     fieldnames = ["filename", "mcq_type1_lighten_orig", "mcq_type1_orig_orig", "mcq_type1_orig_darken"]
#     with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
#         for row in results:
#             writer.writerow(row)

#     print(f"✅ 완료! 저장 위치: {output_file}")

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

# --- 설정값 ---
MAX_WORKERS = 15  # 계정 Tier에 따라 10~20 사이 권장
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
    prompt = USER_PROMPT.format(options=options_text)
    return prompt, num_to_text

# --- API 실행 함수 ---
def run_type1_mcq(img1_path, img2_path):
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
                return TEXT_TO_SCORE[num_to_text[result]]
            time.sleep(0.5 * attempt)
        except Exception:
            time.sleep(1 * attempt)
    return -1.0

# --- 스레드에서 실행할 단위 작업 ---
def process_single_file(fname, dir_orig, dir_lighten, dir_darken):
    path_orig = os.path.join(dir_orig, fname)
    path_lighten = os.path.join(dir_lighten, fname)
    path_darken = os.path.join(dir_darken, fname)

    # 한 파일당 3번의 API 호출 발생
    s_lighten = run_type1_mcq(path_lighten, path_orig)
    s_orig = run_type1_mcq(path_orig, path_orig)
    s_darken = run_type1_mcq(path_orig, path_darken)

    return {
        "filename": fname,
        "mcq_type1_lighten_orig": s_lighten,
        "mcq_type1_orig_orig": s_orig,
        "mcq_type1_orig_darken": s_darken
    }

# --- Main ---
def main():
    base_data_path = "/data1/joo/pai_bench/data/prelim_01"
    dir_orig = os.path.join(base_data_path, "cropped")
    dir_lighten = os.path.join(base_data_path, "cropped_lighten")
    dir_darken = os.path.join(base_data_path, "cropped_darken")

    output_dir = "/data1/joo/pai_bench/result/prelim_01/metric/format/brightness"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "mcq_type1_coarse.csv")

    files = sorted([f for f in os.listdir(dir_orig) if os.path.splitext(f)[1].lower() in VALID_EXT])
    fieldnames = ["filename", "mcq_type1_lighten_orig", "mcq_type1_orig_orig", "mcq_type1_orig_darken"]

    print(f"🚀 총 {len(files)}개 파일 처리 시작 (스레드: {MAX_WORKERS})")

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # ThreadPool 시작
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 작업 등록
            future_to_file = {executor.submit(process_single_file, f, dir_orig, dir_lighten, dir_darken): f for f in files}
            
            completed_count = 0
            for future in as_completed(future_to_file):
                try:
                    result = future.result()
                    writer.writerow(result)
                    csvfile.flush() # 파일에 즉시 기록 (비정상 종료 대비)
                    
                    completed_count += 1
                    if completed_count % 10 == 0:
                        print(f"✅ 진행도: {completed_count}/{len(files)} ({(completed_count/len(files))*100:.1f}%)")
                except Exception as e:
                    fn = future_to_file[future]
                    print(f"❌ {fn} 처리 중 치명적 오류: {e}")

    print(f"✨ 모든 작업 완료! 결과: {output_file}")

if __name__ == "__main__":
    main()