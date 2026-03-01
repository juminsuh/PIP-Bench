import os

def rename_files(target_dir):
    # 해당 경로로 이동
    if not os.path.exists(target_dir):
        print(f"❌ 경로를 찾을 수 없습니다: {target_dir}")
        return

    files = os.listdir(target_dir)
    count = 0

    for filename in files:
        # 확장자 분리
        name, ext = os.path.splitext(filename)
        
        # 파일이 .png이고 이름이 3글자 이상인 경우만 처리
        if ext.lower() == '.jpg' and len(name) >= 3:
            # 앞의 3글자(0xx)만 추출하여 새 파일명 생성
            new_name = name[:3] + ext
            
            old_path = os.path.join(target_dir, filename)
            new_path = os.path.join(target_dir, new_name)

            # 이미 변경된 파일이거나 이름이 같은 경우 스킵
            if filename == new_name:
                continue

            try:
                os.rename(old_path, new_path)
                print(f"✅ Renamed: {filename} -> {new_name}")
                count += 1
            except Exception as e:
                print(f"❌ Error renaming {filename}: {e}")

    print(f"\n총 {count}개의 파일명이 변경되었습니다.")

if __name__ == "__main__":
    target_path = "/data1/joo/pai_bench/data/compare/text"
    rename_files(target_path)