import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 실행 코드가 있는 위치 가져오기
current_dir = os.getcwd()
output_csv = os.path.join(current_dir, "spectral_centroid_analysis.csv")
output_png = os.path.join(current_dir, "relative_difference_spectral_centroid.png")

# 분석할 오디오 파일이 저장된 폴더 경로
real_audio_folder = "/Users/an-yohan/Documents/GitHub/SW/data/newwww/output/ver2/train/real_audio"
fake_audio_folder = "/Users/an-yohan/Documents/GitHub/SW/data/newwww/output/ver2/train/fake_audio"

# 스펙트럼 중심 계산 함수
def calculate_spectral_centroid(audio_folder):
    centroid_values = []
    for file_name in os.listdir(audio_folder):
        file_path = os.path.join(audio_folder, file_name)
        if os.path.isfile(file_path) and file_path.endswith(".ogg"):
            try:
                # 오디오 로드
                y, sr = librosa.load(file_path, sr=32000)

                # 스펙트럼 중심 계산
                spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

                # 파일별 스펙트럼 중심 평균값 저장
                centroid_values.append(spectral_centroid.mean(axis=1)[0])
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
    # 평균과 표준편차 계산
    mean_centroid = np.mean(centroid_values) if centroid_values else 0
    std_centroid = np.std(centroid_values) if centroid_values else 0
    return mean_centroid, std_centroid

# real 데이터 분석
real_mean_centroid, real_std_centroid = calculate_spectral_centroid(real_audio_folder)

# fake 데이터 분석
fake_mean_centroid, fake_std_centroid = calculate_spectral_centroid(fake_audio_folder)

# 상대적 차이 계산
relative_difference = abs(real_mean_centroid - fake_mean_centroid)
relative_std = np.sqrt(real_std_centroid**2 + fake_std_centroid**2)  # 오차 합산

# CSV 저장
results_df = pd.DataFrame({
    "Type": ["Real", "Fake"],
    "Mean_Centroid": [real_mean_centroid, fake_mean_centroid],
    "Std_Centroid": [real_std_centroid, fake_std_centroid]
})
results_df.to_csv(output_csv, index=False)

# 상대적 차이 시각화
plt.figure(figsize=(10, 4))
plt.bar(["Real", "Fake"], [real_mean_centroid, fake_mean_centroid],
        yerr=[real_std_centroid, fake_std_centroid], color=["blue", "red"], alpha=0.7, capsize=5, label="Individual Means")
plt.plot(["Real", "Fake"], [real_mean_centroid, fake_mean_centroid], color="black", linestyle="--", marker="o", label="Trend")
plt.bar(["Relative Difference"], [relative_difference], color="green", alpha=0.7, capsize=5, yerr=[relative_std], label="Relative Difference")
plt.ylabel("Frequency (Hz)")
plt.title(f"Spectral Centroid Analysis\nRelative Difference: {relative_difference:.2f} Hz")
plt.legend()
plt.savefig(output_png)
plt.close()

print(f"Results saved to {output_csv} and {output_png}")
