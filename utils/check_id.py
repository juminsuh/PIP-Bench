import pandas as pd
import os

df = pd.read_csv("./identity_CelebA.csv")
# 2. 저장할 폴더 생성
output_dir = 'ids_grouped'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 3. ID 열을 기준으로 Groupby 실행
grouped = df.groupby('id')

# 4. 각 그룹별로 루프를 돌며 .csv 저장
for identity_id, group in grouped:
    # 파일명 설정 (예: id_1.csv)
    file_name = f"id_{identity_id}.csv"
    file_path = os.path.join(output_dir, file_name)
    
    # 해당 ID에 속한 행들을 csv로 저장
    group.to_csv(file_path, index=False)

print(f"작업 완료! {len(grouped)}개의 ID별 CSV 파일이 '{output_dir}' 폴더에 저장되었습니다.")