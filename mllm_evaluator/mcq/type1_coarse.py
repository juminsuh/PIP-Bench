import os
import base64
import json
import random
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- API 설정 ---
env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

# --- 프롬프트 설정 ---
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
    "They look similar, but are not same people.",
    "They are same people, even they are might under slight different conditions (e.g., lighting, angle, style).",
]

TEXT_TO_SCORE = {
    "They are totally different people.": 0.0,
    "They are quite different people.": 0.25,
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

def get_gen_map(gen_folder):
    """
    gen_folder 내의 파일들(예: 001_xx.jpg)을 스캔하여 { '001': '전체경로' } 맵 생성
    """
    gen_map = {}
    valid_ext = [".jpg", ".jpeg", ".png"]
    if not os.path.exists(gen_folder):
        return gen_map
    
    for fname in os.listdir(gen_folder):
        if os.path.splitext(fname)[1].lower() in valid_ext:
            prefix = fname.split('_')[0]  # '_' 기준 앞부분 추출
            if prefix not in gen_map:
                gen_map[prefix] = os.path.join(gen_folder, fname)
    return gen_map

# --- API ---
def run_type1_mcq(ref_img_path, gen_img_path, user_prompt):
    try:
        ref_b64 = encode_image(ref_img_path)
        gen_b64 = encode_image(gen_img_path)
        ref_mime = get_mime(ref_img_path)
        gen_mime = get_mime(gen_img_path)

        response = client.responses.create(
            model="gpt-5", 
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_prompt},
                        {"type": "input_image", "image_url": f"data:{ref_mime};base64,{ref_b64}"},
                        {"type": "input_image", "image_url": f"data:{gen_mime};base64,{gen_b64}"},
                    ]
                },
            ],
        )
        return response.output_text
    except Exception as e:
        return f"ERROR: {str(e)}"

def process_single_item(idx, ref_path, gen_path):
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        user_prompt, num_to_text = build_shuffled_prompt()
        result = run_type1_mcq(ref_path, gen_path, user_prompt).strip()
        
        if result in num_to_text:
            chosen_text = num_to_text[result]
            return idx, {"text": chosen_text, "score": TEXT_TO_SCORE[chosen_text]}
        
    return idx, {"text": "ERROR", "score": -1}

# --- 메인 실행 함수 ---
def main_with_threading(model, num_workers=15):
    ref_folder = "/data1/joo/pai_bench/data/generation/cropped/orig"
    gen_folder = f"/data1/joo/pai_bench/data/generation/cropped/{model}"
    output_dir = f"/data1/joo/pai_bench/result/mcq/cropped/{model}"
    os.makedirs(output_dir, exist_ok=True)

    gen_map = get_gen_map(gen_folder)
    tasks = []
    results = {}

    valid_ext = [".jpg", ".jpeg", ".png"]
    for fname in sorted(os.listdir(ref_folder)):
        if os.path.splitext(fname)[1].lower() not in valid_ext:
            continue
        
        idx = os.path.splitext(fname)[0] # '001'
        ref_path = os.path.join(ref_folder, fname)
        gen_path = gen_map.get(idx)

        if gen_path:
            tasks.append((idx, ref_path, gen_path))
        else:
            print(f"⚠️ [{idx}] No matching gen image found for model {model}")

    if not tasks:
        print(f"⏩ No tasks to process for {model}.")
        return

    print(f"🚀 Model [{model}]: Processing {len(tasks)} items with {num_workers} workers...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {
            executor.submit(process_single_item, tidx, rpath, gpath): tidx 
            for tidx, rpath, gpath in tasks
        }
        
        for future in as_completed(future_to_idx):
            idx, res = future.result()
            results[idx] = res
            print(f"✅ [{idx}] Done")

    output_list = []
    for tidx in sorted(results.keys()):
        output_list.append({
            "id": tidx.zfill(3),
            "text": results[tidx]["text"],
            "score": results[tidx]["score"]
        })

    output_path = os.path.join(output_dir, "type1_coarse.json")
    with open(output_path, "w") as f:
        json.dump(output_list, f, indent=2, ensure_ascii=False)
    print(f"🏁 Saved: {output_path}\n")

if __name__ == "__main__":
    model_list = ["consistentID", "fastcomposer", "flashface", "gemini", "instantID", "ip_adapter_15_SD"]
    for model in model_list:
        print(f"👉 {model} is processing...")
        main_with_threading(model, num_workers=15)