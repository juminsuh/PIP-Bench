# import json
# import csv
# import os

# def merge_scores_to_csv(mcq_path, human_path, output_path):
#     # 1. mcq 스코어 로드 (딕셔너리 구조)
#     with open(mcq_path, 'r', encoding='utf-8') as f:
#         mcq_data = json.load(f)
#         # "scores" 키 내부의 데이터를 가져옴
#         mcq_dict = mcq_data.get("")

#     # 2. Human 스코어 로드 (리스트 구조)
#     with open(human_path, 'r', encoding='utf-8') as f:
#         human_list = json.load(f)

#     # 3. 데이터 매칭 및 결과 리스트 생성
#     merged_results = []
    
#     for item in human_list:
#         idx = item.get("id")
#         description = item.get("description")
#         human_score = item.get("score")
        
#         # mcq 데이터에서 같은 ID 찾기
#         # zfill(3) 등을 통해 ID 형식을 맞출 필요가 있다면 여기서 처리
#         mcq_info = mcq_dict.get(idx)
        
#         if mcq_info:
#             mcq_score = mcq_info.get("score")
            
#             merged_results.append({
#                 "id": idx,
#                 "description": description.strip(),
#                 "human_score": human_score,
#                 "mcq_score": mcq_score
#             })

#     # 4. CSV 파일로 저장
#     fieldnames = ["id", "description", "human_score", "mcq_score"]
    
#     with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
#         for row in merged_results:
#             writer.writerow(row)

#     print(f"✅ 매칭 완료! 총 {len(merged_results)}개의 행이 저장되었습니다.")
#     print(f"📍 저장 위치: {output_path}")

# if __name__ == "__main__":
#     # 경로 설정
#     mcq_text_path = "/data1/joo/pai_bench/result/prelim_02/type2.json"
#     human_score_path = "/data1/joo/pai_bench/result/prelim_02/human_score.json"
#     output_csv_path = "/data1/joo/pai_bench/result/prelim_02/human_mcq.csv"

#     # 실행
#     merge_scores_to_csv(mcq_text_path, human_score_path, output_csv_path)

import json
import csv
import os

def merge_human_and_mcq_to_csv(human_path, type2_path, output_path):
    # 1. Human 스코어 로드 (리스트 구조)
    with open(human_path, 'r', encoding='utf-8') as f:
        human_list = json.load(f)

    # 2. Type2 MCQ 스코어 로드 (리스트 구조)
    with open(type2_path, 'r', encoding='utf-8') as f:
        type2_list = json.load(f)

    # 3. 효율적인 매칭을 위해 type2 데이터를 딕셔너리로 변환 (ID를 키로 사용)
    # zfill 등을 고려하여 ID를 문자열로 통일
    type2_lookup = {str(item['id']): item['score'] for item in type2_list}

    # 4. 데이터 매칭 및 결과 리스트 생성
    merged_results = []
    
    for item in human_list:
        idx = str(item.get("id"))
        description = item.get("description", "").strip()
        human_score = item.get("score")
        
        # type2_lookup에서 해당 ID의 score 가져오기
        mcq_score = type2_lookup.get(idx)
        
        # 양쪽 파일 모두에 데이터가 있는 경우에만 CSV 행 생성
        if mcq_score is not None:
            merged_results.append({
                "id": idx,
                "description": description,
                "human_score": human_score,
                "mcq_score": mcq_score
            })
        else:
            print(f"⚠️ [Skip] ID {idx} not found in type2.json")

    # 5. CSV 파일로 저장
    fieldnames = ["id", "description", "human_score", "mcq_score"]
    
    # 출력 디렉토리가 없으면 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in merged_results:
            writer.writerow(row)

    print(f"✅ 매칭 및 CSV 생성 완료!")
    print(f"📊 총 데이터 수: {len(merged_results)}개")
    print(f"📍 저장 위치: {output_path}")

if __name__ == "__main__":
    # 경로 설정
    human_score_path = "/data1/joo/pai_bench/result/prelim_02/human_score.json"
    type2_json_path = "/data1/joo/pai_bench/result/prelim_02/type2.json"
    output_csv_path = "/data1/joo/pai_bench/result/prelim_02/human_vs_mcq.csv"

    # 실행
    merge_human_and_mcq_to_csv(human_score_path, type2_json_path, output_csv_path)