import os
import base64
import json
import random
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed


env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

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

# --- API Call Logic ---
def run_type1_mcq(img_path1, img_path2):
    try:
        b64_1 = encode_image(img_path1)
        b64_2 = encode_image(img_path2)
        mime_1 = get_mime(img_path1)
        mime_2 = get_mime(img_path2)

        response = client.chat.completions.create(
            model="gpt-5", 
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": USER_PROMPT},
                        {"type": "input_image", "image_url": f"data:{mime_1};base64,{b64_1}"},
                        {"type": "input_image", "image_url": f"data:{mime_2};base64,{b64_2}"},
                    ]
                },
            ],
            max_tokens=10
        )
        return response.choices[0].message.content
    except Exception:
        return "ERROR"

def process_pair(img1_name, img1_path, img2_name, img2_path):
    img1_fname = img1_name.split(".")[0]
    img2_fname = img2_name.split(".")[0]
    for attempt in range(1, MAX_RETRIES + 1):
        result = run_type1_mcq(img1_path, img2_path).strip()
        if is_valid_score(result):
            return {
                "image0": img1_fname,
                "image1": img2_fname,
                "mcq_type1_score": float(result)
            }
    return {
        "image0": img1_fname,
        "image1": img2_fname,
        "mcq_type1_score": "ERROR"
    }

# --- Main Execution ---
def main():
    orig_folder = "/data1/joo/pai_bench/data/prelim_01/orig"
    output_path = "/data1/joo/pai_bench/result/mcq/prelim_compare/mcq_type1_baseline.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    processed_pairs = set()
    results = []
    if os.path.exists(output_path):
        try:
            with open(output_path, "r") as f:
                results = json.load(f)
                for r in results:
                    if r["mcq_type1_score"] != "ERROR":
                        processed_pairs.add((r["image0"], r["image1"]))
            print(f"🔄 Resuming: {len(processed_pairs)} pairs already processed.")
        except Exception as e:
            print(f"⚠️ Error loading existing file: {e}. Starting fresh.")

    valid_ext = [".jpg", ".jpeg", ".png"]
    img_files = sorted([f for f in os.listdir(orig_folder) if os.path.splitext(f)[1].lower() in valid_ext])
    all_possible_pairs = list(combinations(img_files, 2))
    
    tasks = [p for p in all_possible_pairs if (p[0], p[1]) not in processed_pairs]
    print(f"Total pairs: {len(all_possible_pairs)} | Remaining tasks: {len(tasks)}")

    if not tasks:
        print("✅ All pairs are already processed.")
        return

    CHECKPOINT_INTERVAL = 50
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_pair = {
            executor.submit(
                process_pair, 
                p[0], os.path.join(orig_folder, p[0]), 
                p[1], os.path.join(orig_folder, p[1])
            ): p for p in tasks
        }

        counter = 0
        for future in as_completed(future_to_pair):
            res = future.result()
            results.append(res)
            counter += 1

            if counter % CHECKPOINT_INTERVAL == 0:
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"💾 Checkpoint saved: {len(results)}/{len(all_possible_pairs)} completed.")


    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ All Done! Final count: {len(results)}. Saved → {output_path}")

if __name__ == "__main__":
    main()
