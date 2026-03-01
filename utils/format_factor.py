# import os
# import cv2
# from tqdm import tqdm

# # --- 1) Resize 함수: 비율 유지하며 Small/Big 버전 생성 ---
# def resize_proportionally(img, filename, folder_id, base_output, threshold=512, small_target=256, reg_target=512, big_target=1024):
#     h, w = img.shape[:2]
#     max_side = max(h, w)

#     # 원본이 threshold 이상일 때만 수행
#     if max_side >= threshold:
#         # Small 버전
#         ratio_s = small_target / max_side
#         img_small = cv2.resize(img, (int(w * ratio_s), int(h * ratio_s)), interpolation=cv2.INTER_AREA)
#         # 이미지 저장 직전 크기 출력 코드
#         # print(f"Small: {img_small.shape[1]}x{img_small.shape[0]}")
#         save_path_s = os.path.join(base_output, 'small', folder_id, filename)
#         cv2.imwrite(save_path_s, img_small)
        
#         # regular 버전
#         ratio_r = reg_target / max_side
#         img_regular = cv2.resize(img, (int(w * ratio_r), int(h * ratio_r)), interpolation=cv2.INTER_AREA)
#         # print(f"Regular: {img_regular.shape[1]}x{img_regular.shape[0]}")

#         save_path_r = os.path.join(base_output, 'regular', folder_id, filename)
#         cv2.imwrite(save_path_r, img_regular)
        
#         # # Big 버전
#         ratio_b = big_target / max_side
#         img_big = cv2.resize(img, (int(w * ratio_b), int(h * ratio_b)), interpolation=cv2.INTER_CUBIC)
#         # print(f"Big: {img_big.shape[1]}x{img_big.shape[0]}")

#         save_path_b = os.path.join(base_output, 'big', folder_id, filename)
#         cv2.imwrite(save_path_b, img_big)

# # --- 2) Brightness 함수: 밝게/어둡게 두 버전 생성 ---
# def adjust_brightness(img, filename, folder_id, base_output, value=50):
#     # 밝게 (beta 가 양수)
#     img_lighten = cv2.convertScaleAbs(img, alpha=1.0, beta=value)
#     save_path_l = os.path.join(base_output, 'lighten', folder_id, filename)
#     cv2.imwrite(save_path_l, img_lighten)
    
#     # 어둡게 (beta 가 음수)
#     img_darken = cv2.convertScaleAbs(img, alpha=1.0, beta=-value)
#     save_path_d = os.path.join(base_output, 'darken', folder_id, filename)
#     cv2.imwrite(save_path_d, img_darken)

# # --- 메인 실행 루프 ---
# def run_preprocessing():
#     base_input = '/data1/joo/pai_bench/data/prelim_01/orig'
#     base_output = '/data1/joo/pai_bench/data/prelim_01'
    
#     # 출력 폴더 구조 사전 생성
#     sub_folders = ['small', 'regular', 'big', 'lighten', 'darken']
#     folder_ids = [f"{i:03d}" for i in range(1, 14)]

#     for folder_id in folder_ids:
#         input_path = os.path.join(base_input, folder_id)
#         if not os.path.exists(input_path):
#             continue

#         # 각 모드별 001, 002... 폴더 생성
#         for sub in sub_folders:
#             os.makedirs(os.path.join(base_output, sub, folder_id), exist_ok=True)

#         print(f"\n[Processing Folder: {folder_id}]")
#         images = [f for f in os.listdir(input_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

#         for filename in tqdm(images):
#             img = cv2.imread(os.path.join(input_path, filename))
#             if img is None: continue

#             # 각각의 함수 호출
#             resize_proportionally(img, filename, folder_id, base_output)
#             adjust_brightness(img, filename, folder_id, base_output)

# if __name__ == "__main__":
#     run_preprocessing()

import os
import cv2

# --- 1) Resize 함수: 비율 유지하며 버전 생성 ---
def resize_proportionally(img, filename, base_output, threshold=512, small_target=256, reg_target=512, big_target=1024):
    h, w = img.shape[:2]
    max_side = max(h, w)

    if max_side >= threshold:
        # Small 버전
        ratio_s = small_target / max_side
        img_small = cv2.resize(img, (int(w * ratio_s), int(h * ratio_s)), interpolation=cv2.INTER_AREA)
        save_path_s = os.path.join(base_output, 'small', filename)
        cv2.imwrite(save_path_s, img_small)
        
        # Regular 버전
        ratio_r = reg_target / max_side
        img_regular = cv2.resize(img, (int(w * ratio_r), int(h * ratio_r)), interpolation=cv2.INTER_AREA)
        save_path_r = os.path.join(base_output, 'regular', filename)
        cv2.imwrite(save_path_r, img_regular)
        
        # Big 버전
        ratio_b = big_target / max_side
        img_big = cv2.resize(img, (int(w * ratio_b), int(h * ratio_b)), interpolation=cv2.INTER_CUBIC)
        save_path_b = os.path.join(base_output, 'big', filename)
        cv2.imwrite(save_path_b, img_big)
        print(f"Resize 완료: small, regular, big")

# --- 2) Brightness 함수: 밝게/어둡게 생성 ---
def adjust_brightness(img, filename, base_output, value=50):
    # 밝게
    img_lighten = cv2.convertScaleAbs(img, alpha=1.0, beta=value)
    save_path_l = os.path.join(base_output, 'lighten', filename)
    cv2.imwrite(save_path_l, img_lighten)
    
    # 어둡게
    img_darken = cv2.convertScaleAbs(img, alpha=1.0, beta=-value)
    save_path_d = os.path.join(base_output, 'darken', filename)
    cv2.imwrite(save_path_d, img_darken)
    print(f"밝기 조절 완료: lighten, darken")

def process_single_image(image_path):
    # 설정
    base_output = '/data1/joo/pai_bench/data/prelim_01/'
    filename = os.path.basename(image_path)
    
    # 출력 폴더 구조 생성
    sub_folders = ['small', 'regular', 'big', 'lighten', 'darken']
    for sub in sub_folders:
        os.makedirs(os.path.join(base_output, sub), exist_ok=True)

    # 이미지 읽기
    img = cv2.imread(image_path)
    if img is None:
        print(f"이미지를 불러올 수 없습니다: {image_path}")
        return

    print(f"이미지 처리 중: {filename}")
    
    # 전처리 실행
    resize_proportionally(img, filename, base_output)
    adjust_brightness(img, filename, base_output)
    print("모든 작업이 완료되었습니다.")

if __name__ == "__main__":
    # 처리하고 싶은 파일 경로를 여기에 입력하세요
    target_image_path = '/data1/joo/pai_bench/data/prelim_01/orig/0176.jpg'
    process_single_image(target_image_path)