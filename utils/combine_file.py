import json
import os

def merge_consistent_id_scores(coarse_path, fine_path, output_path):
    # 1. 데이터 로드
    with open(coarse_path, 'r', encoding='utf-8') as f:
        coarse_data = json.load(f)
    with open(fine_path, 'r', encoding='utf-8') as f:
        fine_data = json.load(f)

    # 2. 효율적인 조회를 위해 fine 데이터를 dict로 변환 (id 기준)
    fine_dict = {item['id']: item for item in fine_data}

    merged_results = []

    # 3. Coarse 데이터를 기준으로 병합 진행
    for coarse_item in coarse_data:
        item_id = coarse_item['id']
        
        if item_id in fine_dict:
            fine_item = fine_dict[item_id]
            
            c_score = coarse_item['score']
            f_score = fine_item['score']
            # Score 평균 계산
            if c_score is not None and f_score is not None:
                avg_score = (coarse_item['score'] + fine_item['score']) / 2
                # avg_score = 0.6 * coarse_item['score'] + 0.4 * fine_item['score']
            
                # 형식에 맞춰 데이터 구성
                new_entry = {
                    "id": item_id,
                    "text_coarse": coarse_item.get('text', ""), # 기존 파일의 text 필드명 확인 필요
                    "text_fine": [fine_item.get('text', "")],   # 리스트 형식으로 저장
                    "score": avg_score
                }
                merged_results.append(new_entry)
            else:
                print(f"Warning: ID {item_id} has a None score. (Coarse: {c_score}, Fine: {f_score})")
                continue

    # 4. 결과 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_results, f, indent=2, ensure_ascii=False)

    print(f"Successfully saved merged file to: {output_path}")

# 경로 설정
coarse_file = "/data1/joo/pai_bench/result/mcq/cropped/ip_adapter_15_SD/type1_coarse.json"
fine_file = "/data1/joo/pai_bench/result/mcq/cropped/ip_adapter_15_SD/type1_fine.json"
output_file = "/data1/joo/pai_bench/result/mcq/cropped/ip_adapter_15_SD/type1.json"

# 실행
merge_consistent_id_scores(coarse_file, fine_file, output_file)