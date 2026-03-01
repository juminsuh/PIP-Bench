# import pandas as pd

# # 파일 경로
# easy_pair_path = "/data1/joo/pai_bench/result/prelim_01/pair/easy_pair.csv"
# mcq_type1_sim_path = "/data1/joo/pai_bench/result/prelim_01/metric/content/mcq_type1_baseline.csv"

# # 데이터 로드
# df_easy = pd.read_csv(easy_pair_path)
# df_mcq_type1 = pd.read_csv(mcq_type1_sim_path)

# # img0, img1을 4자리 문자열로 변환
# df_easy['img0'] = df_easy['img0'].apply(lambda x: str(int(x)).zfill(4))
# df_easy['img1'] = df_easy['img1'].apply(lambda x: str(int(x)).zfill(4))

# # mcq_type1 데이터도 4자리 문자열로 변환
# df_mcq_type1['image0'] = df_mcq_type1['image0'].apply(lambda x: str(int(x)).zfill(4))
# df_mcq_type1['image1'] = df_mcq_type1['image1'].apply(lambda x: str(int(x)).zfill(4))

# # mcq_type1 유사도를 딕셔너리로 변환 (빠른 조회를 위해)
# mcq_type1_dict = {}
# for _, row in df_mcq_type1.iterrows():
#     # 양방향으로 저장 (0001-0002와 0002-0001 모두 처리)
#     pair1 = (row['image0'], row['image1'])
#     pair2 = (row['image1'], row['image0'])
#     mcq_type1_dict[pair1] = row['mcq_type1_score']
#     mcq_type1_dict[pair2] = row['mcq_type1_score']

# # easy pair의 mcq_type1 유사도 추출
# easy_mcq_type1_scores = []
# missing_pairs = []

# for _, row in df_easy.iterrows():
#     pair = (row['img0'], row['img1'])
    
#     if pair in mcq_type1_dict:
#         easy_mcq_type1_scores.append(mcq_type1_dict[pair])
#     else:
#         missing_pairs.append(pair)

# # 결과 출력
# print(f"전체 easy pair 개수: {len(df_easy)}")
# print(f"mcq_type1 유사도를 찾은 pair 개수: {len(easy_mcq_type1_scores)}")
# print(f"mcq_type1 유사도를 찾지 못한 pair 개수: {len(missing_pairs)}")

# if easy_mcq_type1_scores:
#     avg_mcq_type1_score = sum(easy_mcq_type1_scores) / len(easy_mcq_type1_scores)
#     print(f"\neasy pair의 평균 mcq_type1 유사도: {avg_mcq_type1_score:.6f}")
# else:
#     print("\n유사도를 계산할 수 없습니다.")

# # 찾지 못한 pair가 있다면 처음 몇 개 출력
# if missing_pairs:
#     print(f"\n찾지 못한 pair 예시 (최대 5개):")
#     for pair in missing_pairs[:5]:
#         print(f"  {pair}")

import pandas as pd

# 파일 경로
easy_pair_path = "/data1/joo/pai_bench/result/prelim_01/pair/easy_pair.csv"
mcq_type1_sim_path = "/data1/joo/pai_bench/result/prelim_01/metric/content/mcq_type1_baseline.csv"

# 데이터 로드
df_easy = pd.read_csv(easy_pair_path)
df_mcq_type1 = pd.read_csv(mcq_type1_sim_path)

# img0, img1을 4자리 문자열로 변환
df_easy['img0'] = df_easy['img0'].apply(lambda x: str(int(x)).zfill(4))
df_easy['img1'] = df_easy['img1'].apply(lambda x: str(int(x)).zfill(4))

# mcq_type1 데이터도 4자리 문자열로 변환
df_mcq_type1['image0'] = df_mcq_type1['image0'].apply(lambda x: str(int(x)).zfill(4))
df_mcq_type1['image1'] = df_mcq_type1['image1'].apply(lambda x: str(int(x)).zfill(4))

# mcq_type1 유사도를 딕셔너리로 변환
mcq_type1_dict = {}
for _, row in df_mcq_type1.iterrows():
    pair1 = (row['image0'], row['image1'])
    pair2 = (row['image1'], row['image0'])
    mcq_type1_dict[pair1] = row['mcq_type1_score']
    mcq_type1_dict[pair2] = row['mcq_type1_score']

# easy pair의 mcq_type1 유사도 추출
easy_mcq_type1_scores = []
missing_pairs = []
error_pairs = [] # "ERROR" 값을 기록하기 위한 리스트

for _, row in df_easy.iterrows():
    pair = (row['img0'], row['img1'])
    
    if pair in mcq_type1_dict:
        val = mcq_type1_dict[pair]
        # --- [수정 포인트] "ERROR"가 아니고 숫자로 변환 가능한 경우만 추가 ---
        if val != "ERROR":
            try:
                easy_mcq_type1_scores.append(float(val))
            except ValueError:
                error_pairs.append(pair)
        else:
            error_pairs.append(pair)
    else:
        missing_pairs.append(pair)

# 결과 출력
print(f"전체 easy pair 개수: {len(df_easy)}")
print(f"숫자 데이터를 찾은 pair 개수: {len(easy_mcq_type1_scores)}")
print(f"ERROR가 발생한 pair 개수: {len(error_pairs)}")
print(f"유사도를 찾지 못한 pair 개수: {len(missing_pairs)}")

# 평균 계산 (숫자 리스트만 사용하므로 에러 없음)
if easy_mcq_type1_scores:
    avg_mcq_type1_score = sum(easy_mcq_type1_scores) / len(easy_mcq_type1_scores)
    print(f"\neasy pair의 평균 mcq_type1 유사도 (ERROR 제외): {avg_mcq_type1_score:.6f}")
else:
    print("\n유사도를 계산할 수 없습니다 (유효한 숫자 데이터 없음).")

# 에러가 발생한 pair가 있다면 예시 출력
if error_pairs:
    print(f"\nERROR 발생 pair 예시 (최대 5개):")
    for pair in error_pairs[:5]:
        print(f"  {pair}")