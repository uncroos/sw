import os
import random

# 데이터가 저장된 폴더 경로
folder_path = '/Users/an-yohan/Documents/GitHub/SW/data/train'

# 폴더 내의 모든 파일 목록을 가져옵니다.
files = os.listdir(folder_path)

# 무작위로 4만 개의 파일을 선택합니다.
files_to_delete = random.sample(files, 40000)

# 선택된 파일을 삭제합니다.
for file in files_to_delete:
    file_path = os.path.join(folder_path, file)
    try:
        os.remove(file_path)
        print(f"삭제됨: {file_path}")
    except Exception as e:
        print(f"파일 삭제 실패: {file_path}, 오류: {e}")

