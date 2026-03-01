import os
import base64
import json
import random
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

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
4.  Do not overestimate or underestimate the decision. Choose the option objectively. 

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

# --- Utils ---
MAX_RETRIES = 3

def is_valid(text, num_to_text):
    parts = [p.strip() for p in text.strip().split(",")]
    return all(p in num_to_text for p in parts)

def parse_response(text, num_to_text):
    parts = [p.strip() for p in text.strip().split(",")]
    chosen_texts = [num_to_text[p] for p in parts]

    if any(t.startswith("Yes") for t in chosen_texts):
        if len(chosen_texts) == 1:
            return chosen_texts, 1.0
        else:
            print("⚠️ 'Yes' option is selected with 'No' options!")
            return None, None
    else:
        num_mistakes = len(chosen_texts)
        score = max(0.0, 1.0 - (1.0 / 6.0) * num_mistakes)
        score = round(score, 4)
        return chosen_texts, score

def build_shuffled_prompt():
    shuffled = OPTIONS.copy()
    random.shuffle(shuffled)
    options_text = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(shuffled))
    num_to_text = {str(i+1): opt for i, opt in enumerate(shuffled)}
    prompt = USER_PROMPT.format(options=options_text)
    return prompt, num_to_text

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_mime(path):
    ext = os.path.splitext(path)[1].lower()
    return "image/png" if ext == ".png" else "image/jpeg"

def get_gen_map(gen_folder):
    gen_map = {}
    valid_ext = [".jpg", ".jpeg", ".png"]
    if not os.path.exists(gen_folder): return gen_map
    for fname in os.listdir(gen_folder):
        if os.path.splitext(fname)[1].lower() in valid_ext:
            prefix = fname.split('_')[0]
            if prefix not in gen_map:
                gen_map[prefix] = os.path.join(gen_folder, fname)
    return gen_map


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
    for attempt in range(1, MAX_RETRIES + 1):
        user_prompt, num_to_text = build_shuffled_prompt()
        result = run_type1_mcq(ref_path, gen_path, user_prompt).strip()
        
        if is_valid(result, num_to_text):
            chosen_texts, chosen_score = parse_response(result, num_to_text)
            if chosen_texts is not None:
                return idx, {"text": chosen_texts, "score": chosen_score}
    
    return idx, {"text": "ERROR", "score": -1}


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
        if os.path.splitext(fname)[1].lower() not in valid_ext: continue
        idx = os.path.splitext(fname)[0]
        
        ref_path = os.path.join(ref_folder, fname)
        gen_path = gen_map.get(idx)

        if gen_path:
            tasks.append((idx, ref_path, gen_path))

    print(f"🚀 Model [{model}]: {len(tasks)} tasks, {num_workers} workers.")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {executor.submit(process_single_item, tidx, rpath, gpath): tidx for tidx, rpath, gpath in tasks}
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

    output_path = os.path.join(output_dir, "type1_fine.json")
    with open(output_path, "w") as f:
        json.dump(output_list, f, indent=2, ensure_ascii=False)
    print(f"🏁 Saved: {output_path}")
        
if __name__ == "__main__":
    model_list = ["consistentID", "fastcomposer", "flashface", "gemini", "instantID", "ip_adapter_15_SD"]
    for model in model_list:
        print(f"👉 {model} is processing...")
        main_with_threading(model, num_workers=15)
