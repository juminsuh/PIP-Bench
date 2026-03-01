import json
import os
import re

def update_gender_prompts(json_path, output_path):
    # 1. JSON 파일 로드
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 2. 정규표현식을 사용하여 정확하게 매칭
    # \b는 단어 경계를 의미하여 'human' 속의 'man' 등을 방지합니다.
    pattern = re.compile(r'\b(a man|a woman)\b', re.IGNORECASE)
    
    for item in data:
        if "prompt" in item:
            # 매칭된 단어 뒤에 토큰 삽입 (예: "a man" -> "a man <|image|>")
            item["prompt"] = pattern.sub(r'\1 <|image|>', item["prompt"])
            
    # 3. 변경된 내용 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"수정 완료! '{output_path}' 파일을 확인하세요.")

# 경로 설정
input_json = "/data1/joo/pai_bench/data/prompts.json"
output_json = "/data1/joo/pai_bench/data/prompts_fastcomposer.json"

update_gender_prompts(input_json, output_json)