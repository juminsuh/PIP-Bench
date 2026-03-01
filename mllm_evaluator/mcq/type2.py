import os
import base64
from openai import OpenAI
import json
from dotenv import load_dotenv
from pathlib import Path
import random

# --- API ---
env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
print(env_path)
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

# --- PROMPT ---
SYSTEM_PROMPT = """
You are a precise vision–language evaluator who inspects whether the text description matches the content in the image based on the specified factors. 
"""

USER_PROMPT = """
[Instruction]
You are given:
	• one image
	• a set of factors: {factors}
	• and a text description: {description}
Your task is to evaluate whether the description correctly matches the visual content of the image specifically in terms of the provided factors. Follow the Rubrics and Actions exactly as written.


[Rubrics]
1.  Factors may include the following elements:
	• action 
	• clothes
	• style (e.g., types of painting/art style)
	• background
2.  Evaluate the image based on the factors included in {factors}.  Only the listed factors should influence your decision.
3.  Ignore any attributes not included in the factor list. If a factor is not included in {factors}, you must not judge it.
4.  Do not overestimate or underestimate the decision. Choose the option objectively. 
 
[Actions]
1. Compare the description **with the image**, evaluating only the factors listed in {factors}. Do not make judgments on any attributes outside these factors.
2. Choose the correct answer from the Options section:
	• If all provided factors match the image → select the option that starts with "Yes".
	• If any provided factor does not match the image → select one or more options that start with "No" corresponding to the mismatched factor(s).
  • You may select multiple options, but **you must not choose an option starting with "Yes" together with any option starting with "No."**
3. Return **only the option number(s)**:
	• If only one option is selected → return a single number (e.g., 1).
	• If more than one options are selected  → return all option numbers separated by commas (e.g., 2, 4, 5)
4. Do not output any explanations, text, or the option sentences. Return only the number(s).


[Options]
{options}
"""


# --- Utils ---

MAX_RETRIES = 3

def is_valid(text, num_to_text):
    parts = [p.strip() for p in text.strip().split(",")]
    return all(p in num_to_text for p in parts)

def parse_response(text, num_to_text, num_factors):
    parts = [p.strip() for p in text.strip().split(",")]
    chosen_texts = [num_to_text[p] for p in parts]

    if any(t.startswith("Yes") for t in chosen_texts):
        if len(chosen_texts) == 1:
            return chosen_texts, 1.0
        else:
            print("  [WARNING] 'Yes' option selected together with 'No' options!")
            return None, None

    num_mistakes = len(chosen_texts)
    score = max(0.0, 1.0 - (1.0 / num_factors) * num_mistakes)
    score = round(score, 4)
    return chosen_texts, score

def build_shuffled_options(class_word, factors):
    yes_option = "Yes, the description accurately describes the image content in terms of all provided factors."
    no_options = [f"No, the {class_word} is not {factor}." for factor in factors]

    all_options = [yes_option] + no_options
    random.shuffle(all_options)

    options_text = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(all_options))
    num_to_text = {str(i+1): opt for i, opt in enumerate(all_options)}
    return options_text, num_to_text

# encode img
def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_mime(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".png":
        return "image/png"
    else:
        return "image/jpeg"
    
VALID_EXT = [".jpg", ".jpeg", ".png"]
# find gen image in subfolder
def find_matching_gen(gen_folder, idx):
    subfolder = os.path.join(gen_folder, idx)
    if not os.path.isdir(subfolder):
        return None
    image_files = [
        fname for fname in os.listdir(subfolder)
        if os.path.splitext(fname)[1].lower() in VALID_EXT
    ]
    if len(image_files) == 0:
        return None
    if len(image_files) > 1:
        print(f"  [WARNING] {idx}/ contains {len(image_files)} images: {sorted(image_files)}. Using first one.")
    return os.path.join(subfolder, sorted(image_files)[0])


def run_type2_mcq(gen_img_path, factors, description, options):
    print("run_type2_mcq called:", gen_img_path)

    gen_b64 = encode_image(gen_img_path)
    gen_mime = get_mime(gen_img_path)

    formatted_prompt = USER_PROMPT.format(factors=factors, description=description, options=options)

    response = client.responses.create(
        model="gpt-5",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": formatted_prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:{gen_mime};base64,{gen_b64}"
                        
                    },
                ]
            },
        ],
    )

    return response.output_text



# --- Run Type2 MCQ evaluation ---
def main(gen_folder_path, parsed_data_path, output_dir):
    gen_folder = gen_folder_path
    
    # load parsed data 
    with open(parsed_data_path, 'r', encoding='utf-8') as f:
        parsed_data = json.load(f)
    
    # create lookup dict for quick access
    parsed_lookup = {item['id']: item for item in parsed_data}


    results = {}
    for folder_name in sorted(os.listdir(gen_folder)):
        subfolder = os.path.join(gen_folder, folder_name)
        if not os.path.isdir(subfolder):
            continue

        idx = folder_name  # "001"
        

        # find image in subfolder
        image_files = [
            f for f in os.listdir(subfolder)
            if os.path.splitext(f)[1].lower() in VALID_EXT
        ]
        if len(image_files) == 0:
            print(f"[{idx}] No image file in subfolder → skip")
            continue
        if len(image_files) > 1:
            print(f"  [WARNING] {idx}/ contains {len(image_files)} images. Using first one.")

        gen_path = os.path.join(subfolder, sorted(image_files)[0])

        # check if parsed data exists for this img
        if idx not in parsed_lookup:
            print(f"[{idx}] No parsed data found → skip")
            continue

        parsed_item = parsed_lookup[idx]
        class_word = parsed_item["class_word"]
        factors = parsed_item['factor']
        description = parsed_item['description']
        num_factors = len(factors)

        print(f"[{idx}] Evaluating... (factors: {factors})")

        try:
            chosen_texts = None
            chosen_score = None

            for attempt in range(1, MAX_RETRIES + 1):
                options_text, num_to_text = build_shuffled_options(class_word, factors)
                result = run_type2_mcq(gen_path, factors, description, options_text)

                if is_valid(result, num_to_text):
                    chosen_texts, chosen_score = parse_response(result, num_to_text, num_factors)
                    if chosen_texts is not None:
                        break
                    else:
                        print(f"  [RETRY {attempt}/{MAX_RETRIES}] Yes+No conflict")
                else:
                    print(f"  [RETRY {attempt}/{MAX_RETRIES}] Invalid response: {result}")

            if chosen_texts is None:
                print(f"  [FAIL] All {MAX_RETRIES} attempts returned invalid responses")
                chosen_texts = ["ERROR"]
                chosen_score = -1

            print(f"[{idx}] → text: {chosen_texts}, score: {chosen_score}")
            results[idx] = {"text": chosen_texts, "score": chosen_score}

        except Exception as e:
            print(f"[{idx}] ERROR: {e}")
            results[idx] = {"text": ["ERROR"], "score": -1}

    # save results
    output_list = []

    for idx, res in results.items():
        output_list.append({
            "id": idx.zfill(3), 
            "text": res["text"],
            "score": res["score"],
        })

    output_path = os.path.join(output_dir, "type2.json")
    
    with open(output_path, "w") as f:
        json.dump(output_list, f, indent=2, ensure_ascii=False)

    print(f"Done! Saved → {output_path}")


if __name__ == "__main__":
    model_list = ["consistentID", "fastcomposer", "flashface", "gemini", "instantID", "ip_adapter_15_SD"]
    for model in model_list:
        print(f"model: {model}")
        gen_folder_path = f"/data1/joo/pai_bench/data/generation/{model}"
        prompt_path = "/data1/joo/pai_bench/data/prompts/prompts.json"
        output_dir = f"/data1/joo/pai_bench/result/mcq/{model}"
        os.makedirs(output_dir, exist_ok=True)
        
        main(gen_folder_path, prompt_path, output_dir)
