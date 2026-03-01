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

# --- Prompt ---
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

# --- API 실행 함수 ---
def run_type1_mcq(img1_path, img2_path):
    ref_b64 = encode_image(img1_path)
    gen_b64 = encode_image(img2_path)

    ref_mime = get_mime(img1_path)
    gen_mime = get_mime(img2_path)
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

                            {
                                "type": "input_image",
                                "image_url": f"data:{ref_mime};base64,{ref_b64}"
                                
                            },
                            {
                                "type": "input_image",
                                "image_url": f"data:{gen_mime};base64,{gen_b64}"
                                
                            },
                        ]
                    },
                ],
            )
            result = response.output_text.strip()
            score = float(result)
            if 0 <= score <= 1:
                return score
        except Exception:
            time.sleep(1 * attempt)
    return -1.0

# --- 스레드에서 실행할 단위 작업 ---
def process_single_file(fname, dir_regular, dir_small, dir_big):
    path_regular = os.path.join(dir_regular, fname)
    path_small = os.path.join(dir_small, fname)
    path_big = os.path.join(dir_big, fname)

    # 한 파일당 3번의 API 호출 발생
    s_small = run_type1_mcq(path_small, path_regular)
    s_regular = run_type1_mcq(path_regular, path_regular)
    s_big = run_type1_mcq(path_regular, path_big)

    return {
        "filename": fname,
        "mcq_type1_small_regular": s_small,
        "mcq_type1_regular_regular": s_regular,
        "mcq_type1_regular_big": s_big
    }

# --- Main ---
def main():
    base_data_path = "/data1/joo/pai_bench/data/prelim_01"
    dir_regular = os.path.join(base_data_path, "cropped_regular")
    dir_small = os.path.join(base_data_path, "cropped_small")
    dir_big = os.path.join(base_data_path, "cropped_big")

    output_dir = "/data1/joo/pai_bench/result/prelim_01/metric/format/size"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "mcq_type1_baseline.csv")

    files = sorted([f for f in os.listdir(dir_regular) if os.path.splitext(f)[1].lower() in VALID_EXT])
    fieldnames = ["filename", "mcq_type1_small_regular", "mcq_type1_regular_regular", "mcq_type1_regular_big"]

    print(f"🚀 총 {len(files)}개 파일 처리 시작 (스레드: {MAX_WORKERS})")

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # ThreadPool 시작
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 작업 등록
            future_to_file = {executor.submit(process_single_file, f, dir_regular, dir_small, dir_big): f for f in files}
            
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