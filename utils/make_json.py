import json
import os

input_path = "/data1/joo/pai_bench/data/prelim_02/prompts.json"
output_path = "/data1/joo/pai_bench/result/prelim_02/human_score.json"

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

result = [{"id": item["id"], "description": item["description"], "score": None} for item in data]

os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"✅ Saved {len(result)} entries to {output_path}")