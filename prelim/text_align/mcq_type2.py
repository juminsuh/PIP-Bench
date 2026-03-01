# import os
# import base64
# from openai import OpenAI
# import json
# from dotenv import load_dotenv
# from pathlib import Path
# import random

# # --- API ---
# env_path = Path(__file__).resolve().parent.parent.parent / '.env'
# load_dotenv(dotenv_path=env_path)
# print(env_path)
# API_KEY = os.getenv("OPENAI_API_KEY")
# client = OpenAI(api_key=API_KEY)

# # --- PROMPT ---
# SYSTEM_PROMPT = """
# You are a precise vision–language evaluator who inspects whether the text description matches the content in the image based on the specified factors. 
# """

# USER_PROMPT = """
# [Instruction]
# You are given:
# 	• one image
# 	• a set of factors: {factors}
# 	• and a text description: {description}
# Your task is to evaluate whether the description correctly matches the visual content of the image specifically in terms of the provided factors. Follow the Rubrics and Actions exactly as written.


# [Rubrics]
# 1.  Factors may include the following elements:
# 	• action 
# 	• clothes
# 	• style (e.g., types of painting/art style)
# 	• background
# 2.  Evaluate the image based on the factors included in {factors}.  Only the listed factors should influence your decision.
# 3.  Ignore any attributes not included in the factor list. If a factor is not included in {factors}, you must not judge it.
# 4.  Do not overestimate or underestimate the decision. Choose the option objectively. 
 
# [Actions]
# 1. Compare the description **with the image**, evaluating only the factors listed in {factors}. Do not make judgments on any attributes outside these factors.
# 2. Choose the correct answer from the Options section:
# 	• If all provided factors match the image → select the option that starts with "Yes".
# 	• If any provided factor does not match the image → select one or more options that start with "No" corresponding to the mismatched factor(s).
#   • You may select multiple options, but **you must not choose an option starting with "Yes" together with any option starting with "No."**
# 3. Return **only the option number(s)**:
# 	• If only one option is selected → return a single number (e.g., 1).
# 	• If more than one options are selected  → return all option numbers separated by commas (e.g., 2, 4, 5)
# 4. Do not output any explanations, text, or the option sentences. Return only the number(s).


# [Options]
# {options}
# """


# # --- Utils ---

# MAX_RETRIES = 3

# def is_valid(text, num_to_text):
#     parts = [p.strip() for p in text.strip().split(",")]
#     return all(p in num_to_text for p in parts)

# def parse_response(text, num_to_text, num_factors):
#     """응답을 파싱하여 선택된 옵션 텍스트 리스트와 점수를 반환"""
#     parts = [p.strip() for p in text.strip().split(",")]
#     chosen_texts = [num_to_text[p] for p in parts]

#     # Yes로 시작하는 옵션이 있으면
#     if any(t.startswith("Yes") for t in chosen_texts):
#         if len(chosen_texts) == 1:
#             return chosen_texts, 1.0
#         else:
#             # Yes와 No가 동시에 선택됨
#             print("  [WARNING] 'Yes' option selected together with 'No' options!")
#             return None, None

#     # No 옵션 개수를 세서 score 계산
#     num_mistakes = len(chosen_texts)
#     score = max(0.0, 1.0 - (1.0 / num_factors) * num_mistakes)
#     score = round(score, 4)
#     return chosen_texts, score

# def build_shuffled_options(class_word, factors):
#     """class_word와 factors로 옵션을 생성하고 셔플, 번호->텍스트 매핑 반환"""
#     yes_option = "Yes, the description accurately describes the image content in terms of all provided factors."
#     no_options = [f"No, the {class_word} is not {factor}." for factor in factors]

#     all_options = [yes_option] + no_options
#     random.shuffle(all_options)

#     options_text = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(all_options))
#     num_to_text = {str(i+1): opt for i, opt in enumerate(all_options)}
#     return options_text, num_to_text

# # encode img
# def encode_image(path):
#     with open(path, "rb") as f:
#         return base64.b64encode(f.read()).decode("utf-8")

# def get_mime(path):
#     ext = os.path.splitext(path)[1].lower()
#     if ext == ".png":
#         return "image/png"
#     else:
#         return "image/jpeg"
    
# VALID_EXT = [".jpg", ".jpeg", ".png"]
# # find gen image in subfolder
# def find_matching_gen(gen_folder, idx):
#     subfolder = os.path.join(gen_folder, idx)
#     if not os.path.isdir(subfolder):
#         return None
#     image_files = [
#         fname for fname in os.listdir(subfolder)
#         if os.path.splitext(fname)[1].lower() in VALID_EXT
#     ]
#     if len(image_files) == 0:
#         return None
#     if len(image_files) > 1:
#         print(f"  [WARNING] {idx}/ contains {len(image_files)} images: {sorted(image_files)}. Using first one.")
#     return os.path.join(subfolder, sorted(image_files)[0])


# def run_type2_mcq(gen_img_path, factors, description, options):
#     print("run_type2_mcq called:", gen_img_path)

#     gen_b64 = encode_image(gen_img_path)
#     gen_mime = get_mime(gen_img_path)

#     formatted_prompt = USER_PROMPT.format(factors=factors, description=description, options=options)

#     response = client.responses.create(
#         model="gpt-5",
#         # temperature=0,
#         # max_tokens=10,
#         input=[
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "input_text", "text": formatted_prompt},
#                     {
#                         "type": "input_image",
#                         "image_url": f"data:{gen_mime};base64,{gen_b64}"
                        
#                     },
#                 ]
#             },
#         ],
#     )

#     return response.output_text



# # --- Run Type2 MCQ evaluation ---
# def main(gen_folder_path, parsed_data_path, output_dir):
#     gen_folder = gen_folder_path
    
#     # load parsed data 
#     with open(parsed_data_path, 'r', encoding='utf-8') as f:
#         parsed_data = json.load(f)
    
#     # create lookup dict for quick access
#     parsed_lookup = {item['id']: item for item in parsed_data}


#     results = {}
#     # for fname in sorted(os.listdir(gen_folder_path)):
#     #     ext = os.path.splitext(fname)[1].lower()

#     #     # 이미지 확장자가 아니면 건너뜀
#     #     if ext not in VALID_EXT:
#     #         continue

#     #     # 파일명에서 확장자를 제외한 부분(ID) 추출
#     #     idx = os.path.splitext(fname)[0]   # 예: "001"
#     #     gen_path = os.path.join(gen_folder_path, fname)
#     for folder_name in sorted(os.listdir(gen_folder)):
#         subfolder = os.path.join(gen_folder, folder_name)
#         if not os.path.isdir(subfolder):
#             continue

#         idx = folder_name  # "001"
        
#         # find image in subfolder
#         image_files = [
#             f for f in os.listdir(subfolder)
#             if os.path.splitext(f)[1].lower() in VALID_EXT
#         ]
#         if len(image_files) == 0:
#             print(f"[{idx}] No image file in subfolder → skip")
#             continue
#         if len(image_files) > 1:
#             print(f"  [WARNING] {idx}/ contains {len(image_files)} images. Using first one.")

#         gen_path = os.path.join(subfolder, sorted(image_files)[0])

#         # check if parsed data exists for this img
#         if idx not in parsed_lookup:
#             print(f"[{idx}] No parsed data found → skip")
#             continue

#         parsed_item = parsed_lookup[idx]
#         class_word = parsed_item["class_word"]
#         factors = parsed_item['factor']
#         description = parsed_item['description']
#         num_factors = len(factors)

#         print(f"[{idx}] Evaluating... (factors: {factors})")

#         try:
#             chosen_texts = None
#             chosen_score = None

#             for attempt in range(1, MAX_RETRIES + 1):
#                 # 매 시도마다 옵션 순서를 새로 셔플
#                 options_text, num_to_text = build_shuffled_options(class_word, factors)
#                 result = run_type2_mcq(gen_path, factors, description, options_text)

#                 if is_valid(result, num_to_text):
#                     chosen_texts, chosen_score = parse_response(result, num_to_text, num_factors)
#                     if chosen_texts is not None:
#                         break
#                     else:
#                         print(f"  [RETRY {attempt}/{MAX_RETRIES}] Yes+No conflict")
#                 else:
#                     print(f"  [RETRY {attempt}/{MAX_RETRIES}] Invalid response: {result}")

#             if chosen_texts is None:
#                 print(f"  [FAIL] All {MAX_RETRIES} attempts returned invalid responses")
#                 chosen_texts = ["ERROR"]
#                 chosen_score = -1

#             print(f"[{idx}] → text: {chosen_texts}, score: {chosen_score}")
#             results[idx] = {"text": chosen_texts, "score": chosen_score}

#         except Exception as e:
#             print(f"[{idx}] ERROR: {e}")
#             results[idx] = {"text": ["ERROR"], "score": -1}

#     # save results
#     output_list = []

#     for idx, res in results.items():
#         output_list.append({
#             "id": idx.zfill(3), 
#             "text": res["text"],
#             "score": res["score"],
#         })

#     output_path = os.path.join(output_dir, "type2.json")
    
#     with open(output_path, "w") as f:
#         json.dump(output_list, f, indent=2, ensure_ascii=False)

#     print(f"Done! Saved → {output_path}")


# if __name__ == "__main__":
#     # model_list = ["consistentID", "fastcomposer", "flashface", "gemini", "instantID", "ip_adapter_15_SD"]
#     # for model in model_list:
#     #     print(f"model: {model}")
#     #     gen_folder_path = f"/data1/joo/pai_bench/data/generation/{model}"
#     #     prompt_path = "/data1/joo/pai_bench/data/prompts/prompts.json"
#     #     output_dir = f"/data1/joo/pai_bench/result/mcq/{model}"
#     #     os.makedirs(output_dir, exist_ok=True)
        
#     #     main(gen_folder_path, prompt_path, output_dir)
#     gen_folder_path = f"/data1/joo/pai_bench/data/prelim_02/images"
#     prompt_path = "/data1/joo/pai_bench/data/prelim_02/prompts.json"
#     output_dir = "/data1/joo/pai_bench/result/prelim_02/type2.json"
#     os.makedirs(output_dir, exist_ok=True)
    
#     main(gen_folder_path, prompt_path, output_dir)
import os
import base64
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

# --- API 설정 ---
env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- 설정 및 상수 ---
MAX_WORKERS = 15 
MAX_RETRIES = 3
VALID_EXT = [".jpg", ".jpeg", ".png"]

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
1. Factors may include the following elements: action, clothes, style, background.
2. Evaluate the image based on the factors included in {factors}.
3. Ignore any attributes not included in the factor list.
4. Do not overestimate or underestimate the decision. Choose the option objectively. 

[Actions]
1. Compare the description with the image, evaluating only the factors listed in {factors}.
2. Choose the correct answer from the Options section.
3. Return only the option number(s).

[Options]
{options}
"""

# --- Utils ---
def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_mime(path):
    ext = os.path.splitext(path)[1].lower()
    return "image/png" if ext == ".png" else "image/jpeg"

def is_valid(text, num_to_text):
    parts = [p.strip() for p in text.strip().split(",")]
    return all(p in num_to_text for p in parts)

def parse_response(text, num_to_text, num_factors):
    parts = [p.strip() for p in text.strip().split(",")]
    chosen_texts = [num_to_text[p] for p in parts]
    if any(t.startswith("Yes") for t in chosen_texts):
        return (chosen_texts, 1.0) if len(chosen_texts) == 1 else (None, None)
    num_mistakes = len(chosen_texts)
    score = round(max(0.0, 1.0 - (1.0 / num_factors) * num_mistakes), 4)
    return chosen_texts, score

def build_shuffled_options(class_word, factors):
    yes_option = "Yes, the description accurately describes the image content in terms of all provided factors."
    no_options = [f"No, the {class_word} is not {factor}." for factor in factors]
    all_opts = [yes_option] + no_options
    random.shuffle(all_opts)
    options_text = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(all_opts))
    num_to_text = {str(i+1): opt for i, opt in enumerate(all_opts)}
    return options_text, num_to_text

# --- 핵심 API 실행 단위 ---
def evaluate_single_item(idx, gen_path, parsed_item):
    class_word = parsed_item["class_word"]
    factors = parsed_item['factor']
    description = parsed_item['description']
    num_factors = len(factors)
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            options_text, num_to_text = build_shuffled_options(class_word, factors)
            formatted_prompt = USER_PROMPT.format(factors=factors, description=description, options=options_text)
            
            response = client.responses.create(
                model="gpt-5",
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": formatted_prompt},
                            {"type": "input_image", "image_url": f"data:{get_mime(gen_path)};base64,{encode_image(gen_path)}"}
                        ]
                    },
                ],
            )
            result = response.output_text.strip()
            if is_valid(result, num_to_text):
                texts, score = parse_response(result, num_to_text, num_factors)
                if texts:
                    return {"id": idx, "text": texts, "score": score}
            time.sleep(0.5 * attempt)
        except Exception as e:
            time.sleep(1 * attempt)
            
    return {"id": idx, "text": ["ERROR"], "score": -1}

# --- Main ---
def main(gen_folder, parsed_data_path, output_dir):
    with open(parsed_data_path, 'r', encoding='utf-8') as f:
        # JSON의 id가 "1" 형태라면 문자열 매칭을 위해 스트립/패딩 처리 고려
        parsed_lookup = {str(item['id']): item for item in json.load(f)}

    tasks = []
    # gen_folder 바로 아래에 있는 파일들을 순회
    for filename in sorted(os.listdir(gen_folder)):
        idx, ext = os.path.splitext(filename)
        if ext.lower() not in VALID_EXT:
            continue
        if idx == "047":
            # 파일명(id)이 parsed_lookup에 있는지 확인
            # (만약 JSON id가 "001"인데 파일명이 "1.png"라면 매칭을 위해 zfill 등이 필요할 수 있음)
            lookup_key = idx # 필요시 idx.lstrip('0') 또는 idx.zfill(3) 등으로 조정
            
            if lookup_key in parsed_lookup:
                gen_path = os.path.join(gen_folder, filename)
                tasks.append((lookup_key, gen_path, parsed_lookup[lookup_key]))

    print(f"🚀 멀티스레딩 평가 시작: 총 {len(tasks)}개 항목 (Workers: {MAX_WORKERS})")
    

    results = []
    output_path = os.path.join(output_dir, "type2.json")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {executor.submit(evaluate_single_item, t[0], t[1], t[2]): t[0] for t in tasks}
        
        count = 0
        for future in as_completed(future_to_idx):
            res = future.result()
            results.append(res)
            count += 1
            print(f"res: {res}")
            
    #         if count % 10 == 0:
    #             print(f"✅ 진행: {count}/{len(tasks)} ({(count/len(tasks))*100:.1f}%)")
    #             with open(output_path, "w", encoding='utf-8') as f:
    #                 json.dump(results, f, indent=2, ensure_ascii=False)

    # # 최종 저장 (ID 순 정렬)
    # results.sort(key=lambda x: x['id'])
    # with open(output_path, "w", encoding='utf-8') as f:
    #     json.dump(results, f, indent=2, ensure_ascii=False)
    
    # print(f"✨ 완료! 저장 위치: {output_path}")

if __name__ == "__main__":
    gen_folder_path = "/data1/joo/pai_bench/data/prelim_02/images"
    prompt_path = "/data1/joo/pai_bench/data/prelim_02/prompts.json"
    output_dir = "/data1/joo/pai_bench/result/prelim_02"
    os.makedirs(output_dir, exist_ok=True)
    
    main(gen_folder_path, prompt_path, output_dir)