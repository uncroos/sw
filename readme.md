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
    pip install torch torchaudio numpy
    pip install demucs

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
