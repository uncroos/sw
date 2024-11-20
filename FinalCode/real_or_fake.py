import os
import pandas as pd
import shutil

# 파일 경로 설정
csv_path = "/Users/an-yohan/Documents/GitHub/SW/data/newwww/new_data/train.csv"  # 원본 CSV 파일 경로
audio_folder = "/Users/an-yohan/Documents/GitHub/SW/data/newwww/new_data/train"   # ogg 파일들이 저장된 폴더
real_csv_path = "/Users/an-yohan/Documents/GitHub/SW/data/newwww/output/ver2/real_csv.csv"      # Real 라벨만 저장할 CSV 파일 경로
fake_csv_path = "/Users/an-yohan/Documents/GitHub/SW/data/newwww/output/ver2/fake_csv.csv"      # Fake 라벨만 저장할 CSV 파일 경로
real_audio_folder = "/Users/an-yohan/Documents/GitHub/SW/data/newwww/output/ver2/real_audio"    # Real 라벨의 오디오 파일 저장 폴더
fake_audio_folder = "/Users/an-yohan/Documents/GitHub/SW/data/newwww/output/ver2/fake_audio"    # Fake 라벨의 오디오 파일 저장 폴더

# 폴더 생성
os.makedirs(real_audio_folder, exist_ok=True)
os.makedirs(fake_audio_folder, exist_ok=True)

# CSV 파일 로드
df = pd.read_csv(csv_path)

# Real과 Fake로 분리
real_df = df[df['label'] == 'real']
fake_df = df[df['label'] == 'fake']

# 분리된 CSV 저장
real_df.to_csv(real_csv_path, index=False)
fake_df.to_csv(fake_csv_path, index=False)

# 오디오 파일 분리 저장
for _, row in real_df.iterrows():
    audio_id = row['id']
    src_path = os.path.join(audio_folder, f"{audio_id}.ogg")
    dst_path = os.path.join(real_audio_folder, f"{audio_id}.ogg")
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)

for _, row in fake_df.iterrows():
    audio_id = row['id']
    src_path = os.path.join(audio_folder, f"{audio_id}.ogg")
    dst_path = os.path.join(fake_audio_folder, f"{audio_id}.ogg")
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)

print("분류 완료!")
