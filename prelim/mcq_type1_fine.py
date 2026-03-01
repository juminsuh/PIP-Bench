import os
import base64
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# --- API 설정 ---
env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- 설정 및 상수 ---
MAX_WORKERS = 20
MAX_RETRIES = 3
VALID_EXT = [".jpg", ".jpeg", ".png"]

SYSTEM_PROMPT = """
You are a powerful visual expert capable of accurately analyzing faces in images and determining whether two people share same fine-grained facial features.
"""

USER_PROMPT = """
[Instruction]
You are given two images: a reference image and a generated image.
Your task is to evaluate whether the two images share same identity-related facial features. Evaluate the Rubrics carefully and follow the Actions exactly. 
Do not output anything other than the option number(s) you select.

[Rubrics]
1. Determine whether the two images depict the same facial features based on:
   • eyes, nose, lips, face shape, skin tone, spatial arrangement of their features
2. Ignore perceptual and coarse-grained impression and focus ONLY on the detailed facial features. 
3. Ignore differences from: lighting, color, posture, angle, expression, hairstyle, makeup, accessories, image quality.

[Actions]
1. Compare identity-related facial features.
2. If the entire facial features are same → select the option starts with "Yes". 
3. If any identity feature differs → select one or more options that start with "No".
4. Return only the option number(s). (e.g., 1 or 2, 4, 5)

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

# --- Utils ---
def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_mime(path):
    ext = os.path.splitext(path)[1].lower()
    return "image/png" if ext == ".png" else "image/jpeg"

def is_valid(text, num_to_text):
    if not text: return False
    parts = [p.strip() for p in text.strip().split(",")]
    return all(p in num_to_text for p in parts)

def parse_response(text, num_to_text):
    parts = [p.strip() for p in text.strip().split(",")]
    chosen_texts = [num_to_text[p] for p in parts]
    
    if any(t.startswith("Yes") for t in chosen_texts):
        if len(chosen_texts) == 1: return chosen_texts, 1.0
        return None, None
    
    num_mistakes = len(chosen_texts)
    score = max(0.0, round(1.0 - (1.0 / 6.0) * num_mistakes, 4))
    return chosen_texts, score

def build_shuffled_prompt():
    shuffled = OPTIONS.copy()
    random.shuffle(shuffled)
    options_text = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(shuffled))
    num_to_text = {str(i+1): opt for i, opt in enumerate(shuffled)}
    prompt = USER_PROMPT.format(options=options_text)
    return prompt, num_to_text

def find_image_path(folder, image_id):
    for ext in VALID_EXT:
        candidate = os.path.join(folder, image_id + ext)
        if os.path.exists(candidate):
            return candidate
    return None

# --- ERROR 항목 재시도 함수 ---
def retry_error_entry(entry, orig_folder_path):
    id_a = entry["id_1"]
    id_b = entry["id_2"]
    
    path_a = find_image_path(orig_folder_path, id_a)
    path_b = find_image_path(orig_folder_path, id_b)
    
    if path_a is None or path_b is None:
        return {"id_1": id_a, "id_2": id_b, "text": "ERROR", "score": -1, "error_msg": "image not found"}
    
    last_err = None
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
                            {"type": "input_image", "image_url": f"data:{get_mime(path_a)};base64,{encode_image(path_a)}"},
                            {"type": "input_image", "image_url": f"data:{get_mime(path_b)};base64,{encode_image(path_b)}"},
                        ]
                    }
                ],
            )
            raw_res = response.output_text.strip() if response.output_text else ""
            
            if is_valid(raw_res, num_to_text):
                texts, score = parse_response(raw_res, num_to_text)
                if texts:
                    return {"id_1": id_a, "id_2": id_b, "text": texts, "score": score}
            
            last_err = f"Invalid response: {raw_res}"
        except Exception as e:
            last_err = str(e)
            time.sleep(1 * attempt)
    
    return {"id_1": id_a, "id_2": id_b, "text": "ERROR", "score": -1, "error_msg": last_err}

# --- Main ---
def main(orig_folder_path, result_json_path):
    # 1. 기존 결과 로드
    if not os.path.exists(result_json_path):
        print(f"❌ Result file not found: {result_json_path}")
        return
    
    with open(result_json_path, "r", encoding='utf-8') as f:
        results = json.load(f)
    
    # 2. ERROR 항목만 필터링
    error_indices = [i for i, entry in enumerate(results) if entry.get("text") == "ERROR"]
    
    if not error_indices:
        print("✅ No ERROR entries found. Nothing to retry!")
        return
    
    print(f"🔄 Found {len(error_indices)} ERROR entries. Retrying with {MAX_WORKERS} workers...")

    # 3. ThreadPoolExecutor로 ERROR 항목만 재시도
    error_entries = [results[i] for i in error_indices]
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(retry_error_entry, entry, orig_folder_path): idx
            for entry, idx in zip(error_entries, error_indices)
        }
        
        completed_count = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            res = future.result()
            results[idx] = res
            completed_count += 1
            
            if completed_count % 50 == 0 or completed_count == len(error_indices):
                print(f"✅ Progress: {completed_count}/{len(error_indices)} ({(completed_count/len(error_indices))*100:.2f}%)")
                with open(result_json_path, "w", encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

    # 4. 최종 저장
    with open(result_json_path, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 5. 결과 요약
    remaining_errors = sum(1 for entry in results if entry.get("text") == "ERROR")
    print(f"✨ Retry finished! {len(error_indices) - remaining_errors}/{len(error_indices)} errors resolved.")
    if remaining_errors > 0:
        print(f"⚠️ {remaining_errors} entries still have errors.")

if __name__ == "__main__":
    orig_folder = "/data1/joo/pai_bench/data/prelim_01/orig"
    result_json = "/data1/joo/pai_bench/result/mcq/prelim_compare/type1_fine.json"
    
    main(orig_folder, result_json)