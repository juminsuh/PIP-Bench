import os
import base64
from openai import OpenAI
import json
from dotenv import load_dotenv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed # 멀티스레딩 추가

# --- API ---
env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)


# --- PROMPT ---
SYSTEM_PROMPT = """
You are a precise vision–language evaluator who inspects whether the text description matches the content in the image based on the text description. 
"""

USER_PROMPT = """
[Instruction]
You are given:
    • one image
    • and a text description: {description}
Your task is to evaluate whether the description correctly matches the visual content of the image. Follow the Rubrics and Actions exactly as written.

[Rubrics]
1.  Evaluate the image based on the description.
2.  Do not overestimate or underestimate the score. Assign the score objectively. 
3.  Ignore any attributes not included in the description.

[Actions]
1. Compare the description **with the image**. 
2. Rate how well the image matches the text description on a scale from 0 (no match) to 1 (perfect match).
3. Return **only the rated number(s)**.
4. Do not output any explanations, text, or the option sentences. Return only the number(s).
"""


# --- Utils ---
MAX_RETRIES = 3
VALID_EXT = [".jpg", ".jpeg", ".png"]

def is_valid_score(text):
    try:
        score = float(text.strip())
        return 0 <= score <= 1
    except ValueError:
        return False
    
def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_mime(path):
    ext = os.path.splitext(path)[1].lower()
    return "image/png" if ext == ".png" else "image/jpeg"


# --- Type2_MCQ ---
def run_type2_mcq(gen_img_path, description):
    gen_b64 = encode_image(gen_img_path)
    gen_mime = get_mime(gen_img_path)
    formatted_prompt = USER_PROMPT.format(description=description)

    response = client.responses.create(
        model="gpt-5",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": formatted_prompt},
                    {"type": "input_image", "image_url": f"data:{gen_mime};base64,{gen_b64}"},
                ]
            },
        ],
    )
    return response.output_text


# --- Threading ---
def process_single_item(idx, gen_path, description):
    print(f"➡️ [{idx}] Evaluating...")
    try:
        mcq_out = None
        for attempt in range(1, MAX_RETRIES + 1):
            result = run_type2_mcq(gen_path, description)
            if is_valid_score(result):
                mcq_out = result.strip()
                break
            else:
                print(f"  [RETRY {attempt}/{MAX_RETRIES}] {idx}: Invalid response")

        if mcq_out is None:
            print(f"  [FAIL] {idx}: All attempts failed")
            mcq_out = "ERROR"
            
        return idx, mcq_out

    except Exception as e:
        print(f"[{idx}] ERROR: {e}")
        return idx, "ERROR"


# --- Run Type2 MCQ evaluation ---
def main(gen_folder_path, prompt_path, output_dir, max_workers=15):
    os.makedirs(output_dir, exist_ok=True)
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_data = json.load(f)
    lookup = {item['id']: item['description'] for item in prompt_data}

    tasks = []
    for folder_name in sorted(os.listdir(gen_folder_path)):
        subfolder = os.path.join(gen_folder_path, folder_name)
        if not os.path.isdir(subfolder):
            continue
            
        idx = folder_name
        image_files = [
            f for f in os.listdir(subfolder)
            if os.path.splitext(f)[1].lower() in VALID_EXT
        ]
        
        if not image_files:
            continue
        if idx not in lookup:
            continue

        gen_path = os.path.join(subfolder, sorted(image_files)[0])
        description = lookup[idx]
        tasks.append((idx, gen_path, description))

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(process_single_item, idx, path, desc): idx 
            for idx, path, desc in tasks
        }

        for future in as_completed(future_to_idx):
            idx, mcq_out = future.result()
            results[idx] = mcq_out
            print(f"👉 [{idx}] → {mcq_out}")

    output_list = []
    for idx in sorted(results.keys()):
        output_list.append({
            "id": idx.zfill(3), 
            "result": results[idx]
        })

    output_path = os.path.join(output_dir, "type2_baseline.json")
    with open(output_path, "w") as f:
        json.dump(output_list, f, indent=2, ensure_ascii=False)

    print(f"✨ Done! Saved → {output_path}")


if __name__ == "__main__":
    model_list = ["consistentID", "fastcomposer", "flashface", "gemini", "instantID", "ip_adapter_15_SD"]
    
    for model in model_list:
        print(f"\n🚀 Current Model: {model}")
        gen_folder = f"/data1/joo/pai_bench/data/generation/{model}"
        prompt_file = "/data1/joo/pai_bench/data/prompts/prompts.json"
        out_dir = f"/data1/joo/pai_bench/result/mcq/{model}"
        
        main(gen_folder, prompt_file, out_dir, max_workers=15)