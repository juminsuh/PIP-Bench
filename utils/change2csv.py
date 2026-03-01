import pandas as pd
import json

# 1. JSON 파일 불러오기
# 파일 경로를 입력하세요 (예: 'data.json')
file_path = '/data1/joo/pai_bench/result/mcq/prelim_compare/mcq_type1_baseline.json'

with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. 데이터프레임 생성
df = pd.DataFrame(data)

# 3. 열 이름 변경 및 필요한 열만 선택
# 만약 JSON의 키값이 정확히 id_1, id_2, score라면 아래와 같이 매핑합니다.
df_filtered = df[['image0', 'image1', 'mcq_type1_score']].rename(columns={
    'image0': 'image0',
    'image0': 'image1',
    'mcq_type1_score': 'mcq_type1_score'
})

# 4. CSV 파일로 저장
output_path = '/data1/joo/pai_bench/result/prelim_01/metric/content/mcq_type1_baseline.csv'
df_filtered.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"변환 완료! 파일이 {output_path}로 저장되었습니다.")