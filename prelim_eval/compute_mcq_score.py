import json

def calculate_average_result(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # result 값들만 추출 (데이터가 리스트 형태인 경우)
        results_str = [item['result'] for item in data if 'result' in item]
        results = [float(score) for score in results_str if score != 'ERROR']
        
        if not results:
            print("파일에 result 데이터가 없습니다.")
            return None
        
        average = sum(results) / len(results)
        print(f"총 {len(results)}개의 데이터에 대한 평균 점수: {average:.4f}")
        return average

    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {file_path}")
    except json.JSONDecodeError:
        print(f"JSON 형식이 올바르지 않습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")

# 경로 설정
file_path = "/data1/joo/pai_bench/result/mcq/ip_adapter_15_SD/type2_baseline.json"

# 실행
calculate_average_result(file_path)