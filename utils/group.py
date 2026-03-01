import json 
from pathlib import Path
from collections import defaultdict

def group_images_by_id(metadata_list):
    results = []
    
    for i in range(0, len(metadata_list), 15):
        id_metadata = metadata_list[i:i+15]
        
        if not id_metadata:
            break

        groups = {
            "facial_expression": defaultdict(list),
            "angle": defaultdict(list),
            "style": defaultdict(list)
        }
        
        for meta in id_metadata:
            img_id = meta.get("img_id")
            
            # 1. Facial Expression
            expr = meta.get("facial_expression")
            if expr in ["big smile", "light smile", "neutral", "surprised"]:
                groups["facial_expression"][expr].append(img_id)
            
            # 2. Angle 
            ang = meta.get("angle")
            if ang in ["left", "straight", "right"]:
                groups["angle"][ang].append(img_id)
                
            # 3. Style
            style_key = f"{meta.get('hair_color')}_{meta.get('mustache')}_{meta.get('occlusion')}"
            groups["style"][style_key].append(img_id)
            
        results.append({
            "id_index": i // 15 + 1,
            "groups": groups
        })
        
    return results

def main():
    metadata_dir = "/data1/joo/pai_bench/data/prelim_01/metadata.jsonl"
    metadata_list = []
    
    if not Path(metadata_dir).exists():
        print(f"❌ 파일을 찾을 수 없습니다: {metadata_dir}")
        return

    with open(metadata_dir, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip() 
            if not line:     
                continue
            try:
                data = json.loads(line)
                metadata_list.append(data)
            except json.JSONDecodeError as e:
                print(f"⚠️ {line_num}행에서 형식 오류 발생: {e}")
                continue
            
    if not metadata_list:
        print("로드된 데이터가 없습니다.")
        return

    grouped_data = group_images_by_id(metadata_list=metadata_list)

    # 출력 및 저장
    output_path = "/data1/joo/pai_bench/data/prelim_01/grouping_results.json"
    
    print("--- 그룹화 결과 요약 ---")
    output_json = json.dumps(grouped_data, indent=4, ensure_ascii=False)
    print(output_json)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output_json)

    print(f"\n✅ 결과가 저장되었습니다: {output_path}")
    
if __name__ == "__main__":
    main()