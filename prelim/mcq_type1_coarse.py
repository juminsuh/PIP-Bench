import os
import base64
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAX_WORKERS = 20
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
3. Ignore differences from: lighting, color, posture, angle, expression, hairstyle, makeup, accessories, image quality.

[Actions]
1. Compare perceptual impressions of two images.
2. Select the option that best describes the identity match. 
3. Return ONLY the single number.

[Options]
{options}
"""

OPTIONS = [
    "They are totally different people.",
    "They are quite different people.",
    "They look similar, but are not same people.",
    "They are same people, even they are might under slight different conditions (e.g., lighting, angle, style).",
]

TEXT_TO_SCORE = {
    "They are totally different people.": 0.0,
    "They are quite different people.": 0.25,
    "They look similar, but are not same people.": 0.75,
    "They are same people, even they are might under slight different conditions (e.g., lighting, angle, style).": 1.0,
}

# --- 유틸리티 함수 ---
def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_mime(path):
    ext = os.path.splitext(path)[1].lower()
    return "image/png" if ext == ".png" else "image/jpeg"

def is_valid(text, num_to_text):
    return text.strip() in num_to_text

def build_shuffled_prompt():
    shuffled = OPTIONS.copy()
    random.shuffle(shuffled)
    options_text = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(shuffled))
    num_to_text = {str(i+1): opt for i, opt in enumerate(shuffled)}
    prompt = USER_PROMPT.format(options=options_text)
    return prompt, num_to_text

def retry_error_entry(entry, orig_folder_path):
    id_a = entry["id_1"]
    id_b = entry["id_2"]
    
    path_a = find_image_path(orig_folder_path, id_a)
    path_b = find_image_path(orig_folder_path, id_b)
    
    if path_a is None or path_b is None:
        return {"id_1": id_a, "id_2": id_b, "text": "ERROR: image not found", "score": -1}
    
    try:
        chosen_text, chosen_score = None, None
        for attempt in range(1, MAX_RETRIES + 1):
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
            
            result = response.output_text
            if is_valid(result, num_to_text):
                chosen_text = num_to_text[result]
                chosen_score = TEXT_TO_SCORE[chosen_text]
                break
        
        if not chosen_text:
            return {"id_1": id_a, "id_2": id_b, "text": "ERROR", "score": -1}
        
        return {"id_1": id_a, "id_2": id_b, "text": chosen_text, "score": chosen_score}
    
    except Exception as e:
        return {"id_1": id_a, "id_2": id_b, "text": f"EXCEPTION: {str(e)}", "score": -1}

def find_image_path(folder, image_id):
    for ext in VALID_EXT:
        candidate = os.path.join(folder, image_id + ext)
        if os.path.exists(candidate):
            return candidate
    return None

def main(orig_folder_path, result_json_path):
    if not os.path.exists(result_json_path):
        print(f"❌ Result file not found: {result_json_path}")
        return
    
    with open(result_json_path, "r", encoding='utf-8') as f:
        results = json.load(f)
    
    error_indices = [i for i, entry in enumerate(results) if entry.get("text") == "ERROR"]
    
    if not error_indices:
        print("✅ No ERROR entries found. Nothing to retry!")
        return
    
    print(f"🔄 Found {len(error_indices)} ERROR entries. Retrying with {MAX_WORKERS} workers...")

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

    with open(result_json_path, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 5. 결과 요약
    remaining_errors = sum(1 for entry in results if entry.get("text") == "ERROR" or entry.get("text", "").startswith("EXCEPTION"))
    print(f"✨ Retry finished! {len(error_indices) - remaining_errors}/{len(error_indices)} errors resolved.")
    if remaining_errors > 0:
        print(f"⚠️ {remaining_errors} entries still have errors.")

if __name__ == "__main__":
    orig_folder = "/data1/joo/pai_bench/data/prelim_01/orig"
    result_json = "/data1/joo/pai_bench/result/mcq/prelim_compare/type1_coarse.json"
    
    main(orig_folder, result_json)
