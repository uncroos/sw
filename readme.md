# 가상환경

- 가상환경 설치

  ```
  python3 -m venv venv
  ```

- 가상환경 활성화
  ```
  source venv/bin/activate
  ```

# 라이브러리 설치

    pip install pandas
    pip install matplotlib
    pip install seaborn
    pip install librosa
    pip install torch
    pip install torchaudio
    pip install numpy
    pip install demucs
    pip install scikit-learn
    pip install joblib

# 구조 설명

1. `ModifyingTrainData` 안에 있는 모든 py파일을 실행한다

- `ModifyingTrainData`는 기본적인 데이터 전처리 파일이다
- 이 폴더에서는 최종적으로 `train_ogg_noise_remove.py`를 실행하여 train데이터의 노이즈를 제거하고, 5초로 규격화 시킨다.

2. `FileOrganization` 안에 있는 모든 py파일을 실행한다.

- `FileOrganization`의 파일은 5만여개의 데이터를 가진 train과 test 폴더를 1만개까지 램덤으로 지운다.

3. `FinalCode`

- ver1 폴더는 말그대로 버전 1이기 때문에 상관이 없다.
- ver2 폴더가 최종 폴더이다.
- ver2 폴더에서 `real_or_fake.py`를 실행시켜 1만여개의 데이터를 real데이터 혹은 fake데이터로 나눈다.
- 그 후, `real.py`과 `fake.py`파일에서 SpectralCentroid에 대한 그래프를 그린다.

# 파일 구조

```
.
├── FileOrganization
│   ├── py_1.py
│   ├── py_2.py
│   └── py_3.py
├── FinalCode
│   ├── model.py
│   ├── new_data
│   │   ├── Number_of_files.py
│   │   ├── test.csv
│   │   └── train.csv
│   ├── output
│   │   ├── ver1
│   │   │   ├── file
│   │   │   │   ├── submission.csv
│   │   │   │   └── submission_with_result.csv
│   │   │   ├── image
│   │   │   │   ├── label_distribution.png
│   │   │   │   ├── mfcc_visualization.png
│   │   │   │   ├── spectral_centroid_visualization.png
│   │   │   │   └── spectrogram_visualization.png
│   │   │   └── result.py
│   │   └── ver2
│   │       ├── SpectralCentroid
│   │       │   ├── file
│   │       │   │   ├── fake_spectral_centroid_mean.csv
│   │       │   │   └── real_spectral_centroid_mean.csv
│   │       │   └── image
│   │       │       ├── fake_spectral_centroid_distribution.png
│   │       │       └── real_spectral_centroid_distribution.png
│   │       ├── code
│   │       │   ├── fake.py
│   │       │   └── real.py
│   │       └── train
│   │           ├── fake_csv.csv
│   │           └── real_csv.csv
│   └── real_or_fake.py
└── ModifyingTrainData
    ├── image
    │   ├── audio_length_density.png
    │   └── average_duration_bar.png
    ├── new_data
    │   ├── train_sorted.csv
    │   └── train_time.csv
    ├── tarin_average.py
    ├── train.py
    ├── train_data_1.py
    ├── train_ogg_noise_remove.py
    ├── train_same_time.py
    └── train_time.py
```
