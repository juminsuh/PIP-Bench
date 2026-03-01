import os
import base64
from openai import OpenAI
import json
from dotenv import load_dotenv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)


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
MAX_RETRIES = 3
VALID_EXT = [".jpg", ".jpeg", ".png"]

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def is_valid_score(text):
    try:
        score = float(text.strip())
        return 0 <= score <= 1
    except ValueError:
        return False
    
def get_mime(path):
    ext = os.path.splitext(path)[1].lower()
    return "image/png" if ext == ".png" else "image/jpeg"

def find_matching_gen(gen_folder, idx):
   
    if not os.path.isdir(gen_folder):
        return None
        
    for fname in os.listdir(gen_folder):
        if os.path.splitext(fname)[1].lower() in VALID_EXT:
            gen_idx = fname.split('_')[0]
            
            if gen_idx == idx:
                return os.path.join(gen_folder, fname)
                
    return None


# --- Type1_MCQ  ---
def run_type1_mcq(ref_img_path, gen_img_path):
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
                    {"type": "input_text", "text": USER_PROMPT},
                    {"type": "input_image", "image_url": f"data:{ref_mime};base64,{ref_b64}"},
                    {"type": "input_image", "image_url": f"data:{gen_mime};base64,{gen_b64}"},
                ]
            },
        ],
    )
    return response.output_text



def process_single_id(idx, ref_path, gen_path):
    print(f"[{idx}] Evaluating... (Matching: {os.path.basename(gen_path)})")
    try:
        mcq_out = None
        for attempt in range(1, MAX_RETRIES + 1):
            result = run_type1_mcq(ref_path, gen_path)
            if is_valid_score(result):
                mcq_out = result.strip()
                break
            else:
                print(f"  [RETRY {attempt}/{MAX_RETRIES}] {idx}: Invalid response")

        if mcq_out is None:
            mcq_out = "ERROR"
        
        return idx, mcq_out

    except Exception as e:
        print(f"[{idx}] ERROR: {e}")
        return idx, "ERROR"


# --- Main ---
def main(ref_folder_path, gen_folder_path, output_dir, max_workers=15):
    os.makedirs(output_dir, exist_ok=True)

    tasks = []
    for fname in sorted(os.listdir(ref_folder_path)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in VALID_EXT:
            continue

        idx = os.path.splitext(fname)[0]
        ref_path = os.path.join(ref_folder_path, fname)
        
        gen_path = find_matching_gen(gen_folder_path, idx)

        if gen_path is None:
            print(f"[{idx}] No matching generated file (id_*.jpg) in gen_folder → skip")
            continue
        
        tasks.append((idx, ref_path, gen_path))

    print(f"🚀 Total tasks identified: {len(tasks)}")

    final_results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(process_single_id, idx, ref, gen): idx 
            for idx, ref, gen in tasks
        }

        for future in as_completed(future_to_idx):
            idx, res = future.result()
            final_results[idx] = res
            print(f"✅ [{idx}] Result: {res}")

    output_list = []
    for idx in sorted(final_results.keys()):
        output_list.append({
            "id": idx.zfill(3), 
            "result": final_results[idx]
        })

    output_path = os.path.join(output_dir, "type1_baseline.json")
    with open(output_path, "w") as f:
        json.dump(output_list, f, indent=2, ensure_ascii=False)

    print(f"\n✨ Done! Processed {len(output_list)} pairs. Saved → {output_path}")


if __name__ == "__main__":
    for model in ["consistentID", "fastcomposer", "flashface", "gemini", "instantID"]:

        ref_folder = "/data1/joo/pai_bench/data/generation/cropped/ablation/ref"
        gen_folder = f"/data1/joo/pai_bench/data/generation/cropped/ablation/{model}"
        output_dir = f"/data1/joo/pai_bench/result/mcq/cropped/{model}"
        
        main(ref_folder, gen_folder, output_dir, max_workers=15)
