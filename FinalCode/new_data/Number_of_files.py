import os

# 데이터가 저장된 폴더 경로
folder_path = '/Users/an-yohan/Documents/GitHub/SW/data/new_data/test'

# 폴더 내의 파일 목록을 가져오고 파일 수를 셉니다.
file_count = len([file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))])

print(f"폴더 내 파일 수: {file_count}")
