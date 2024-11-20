import os
from demucs import pretrained
from demucs.separate import load_track
import torchaudio

# 경로 설정
source_dir = "/Users/an-yohan/Documents/GitHub/SW/data/test"          # 분석할 데이터 폴더
output_dir = "/Users/an-yohan/Documents/GitHub/SW/data/new_test_ogg"   # 저장할 폴더

# 저장 폴더가 없으면 생성
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Demucs 모델 불러오기
model = pretrained.get_model(name="htdemucs")  # 사전 학습된 모델 사용

# 오디오 파일 노이즈 제거 및 저장
for file_name in sorted(os.listdir(source_dir)):
    if file_name.endswith(".ogg") and "TEST_" in file_name:
        file_path = os.path.join(source_dir, file_name)
        print(f"Processing {file_name}...")

        # 오디오 파일 불러오기
        waveform, sample_rate = load_track(file_path)

        # 소음 제거
        sources = model.apply(waveform, sample_rate=sample_rate)

        # 결과물 저장 (목소리만 저장)
        vocals = sources["vocals"]  # 'vocals' 채널에서 음성 분리
        output_path = os.path.join(output_dir, file_name)
        torchaudio.save(output_path, vocals, sample_rate=sample_rate)
        print(f"Saved processed file: {output_path}")
