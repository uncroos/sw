import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 데이터 로드 및 설정
train_csv_path = "/Users/an-yohan/Documents/GitHub/SW/data/new_data/train.csv"
test_csv_path = "/Users/an-yohan/Documents/GitHub/SW/data/new_data/test.csv"
train_folder = "/Users/an-yohan/Documents/GitHub/SW/data/new_data/train"
test_folder = "/Users/an-yohan/Documents/GitHub/SW/data/new_data/test"
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# 주파수 특징 추출 함수
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=32000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)  # MFCC 특징
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features = np.hstack([
            mfcc.mean(axis=1),
            spectral_centroid.mean(axis=1),
            spectral_bandwidth.mean(axis=1)
        ])
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# 학습 데이터 로드
train_data = pd.read_csv(train_csv_path)
train_data["path"] = train_data["path"].apply(lambda x: x.replace('./train/', 'train/'))
train_data["features"] = train_data["path"].apply(extract_features)

# 특징이 없는 데이터 제거
train_data = train_data.dropna(subset=["features"])
X = np.vstack(train_data["features"].values)
y = train_data["label"].map({"real": 1, "fake": 0})

# 학습/검증 데이터 분리
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 검증 데이터 예측
y_pred = clf.predict(X_val)
print("Validation Report:\n", classification_report(y_val, y_pred))

# 테스트 데이터 처리 및 예측
test_data = pd.read_csv(test_csv_path)
test_data["path"] = test_data["path"].apply(lambda x: x.replace('./test/', 'test/'))
test_data["features"] = test_data["path"].apply(extract_features)

test_data = test_data.dropna(subset=["features"])
X_test = np.vstack(test_data["features"].values)
predictions = clf.predict_proba(X_test)

# 제출 파일 생성
submission = pd.DataFrame({
    "id": test_data["id"],
    "fake": predictions[:, 0],
    "real": predictions[:, 1]
})
submission.to_csv(os.path.join(output_folder, "submission.csv"), index=False)
print("Submission saved.")

# 시각화
# 1. MFCC 시각화
sample_file = train_data.iloc[0]["path"]
y, sr = librosa.load(sample_file, sr=32000)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc, x_axis="time")
plt.colorbar()
plt.title("MFCC of Sample Audio")
plt.savefig(os.path.join(output_folder, "mfcc_visualization.png"))
plt.close()

# 2. 스펙트럼 시각화
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
plt.figure(figsize=(10, 4))
librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log")
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram of Sample Audio")
plt.savefig(os.path.join(output_folder, "spectrogram_visualization.png"))
plt.close()

# 3. 스펙트럼 중심 시각화
spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
frames = range(len(spectral_centroid[0]))
times = librosa.frames_to_time(frames, sr=sr)
plt.figure(figsize=(10, 4))
plt.semilogy(times, spectral_centroid.T, label="Spectral Centroid")
plt.xlabel("Time (s)")
plt.ylabel("Hz")
plt.title("Spectral Centroid of Sample Audio")
plt.legend()
plt.savefig(os.path.join(output_folder, "spectral_centroid_visualization.png"))
plt.close()

# 4. 데이터 분포 시각화
real_count = train_data[train_data["label"] == "real"].shape[0]
fake_count = train_data[train_data["label"] == "fake"].shape[0]
plt.figure(figsize=(6, 6))
plt.pie([real_count, fake_count], labels=["Real", "Fake"], autopct="%1.1f%%", startangle=90)
plt.title("Label Distribution in Training Data")
plt.savefig(os.path.join(output_folder, "label_distribution.png"))
plt.close()
