import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

# 경로 설정
real_audio_folder = "/Users/an-yohan/Documents/GitHub/SW/data/newwww/output/ver2/train/real_audio"
output_folder = "/Users/an-yohan/Documents/GitHub/SW/data/newwww/output/ver2/SpectralCentroid"
output_csv = os.path.join(output_folder, "real_spectral_centroid_mean.csv")

# 결과 저장 폴더 생성
os.makedirs(output_folder, exist_ok=True)

# Spectral Centroid 평균값 저장 리스트
spectral_centroids = []

# Real 오디오 파일 처리
for file_name in os.listdir(real_audio_folder):
    if file_name.endswith(".ogg"):  # 오디오 파일 확장자
        file_path = os.path.join(real_audio_folder, file_name)
        
        # 오디오 로드
        y, sr = librosa.load(file_path, sr=None)
        
        # Spectral Centroid 계산
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroids.append(spectral_centroid.mean())  # 평균값 저장

# 전체 평균값 계산
overall_mean_centroid = np.mean(spectral_centroids)

# 결과 저장 (CSV)
with open(output_csv, "w") as f:
    f.write("Overall_Mean_Spectral_Centroid\n")
    f.write(f"{overall_mean_centroid}\n")

# 시각화 (히스토그램)
plt.figure(figsize=(10, 6))
plt.hist(spectral_centroids, bins=20, color="skyblue", edgecolor="black")
plt.axvline(overall_mean_centroid, color="red", linestyle="--", label=f"Overall Mean: {overall_mean_centroid:.2f} Hz")
plt.xlabel("Spectral Centroid (Hz)")
plt.ylabel("Frequency")
plt.title("Distribution of Spectral Centroids (Real Audio)")
plt.legend()
plt.grid(True)

# 시각화 결과 저장
visualization_path = os.path.join(output_folder, "real_spectral_centroid_distribution.png")
plt.savefig(visualization_path)
plt.close()

print(f"Spectral Centroid의 전체 평균값이 {output_csv}에 저장되었습니다.")
print(f"히스토그램 시각화가 {visualization_path}에 저장되었습니다.")
