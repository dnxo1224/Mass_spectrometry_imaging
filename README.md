# 🔬 Mass Spectrometry Imaging — Super Resolution

ToF-SIMS (Time-of-Flight Secondary Ion Mass Spectrometry) 이미지의 **초해상도(Super Resolution)** 복원 파이프라인입니다.  
저해상도(Low Resolution) 질량 이미지를 딥러닝 기반 모델(SRCNN, SRGAN)로 고해상도(High Resolution)로 변환합니다.

---

## 📁 프로젝트 구조

```
Mass_spectrometry_imaging/
├── PythonFile/                         # 메인 소스 코드
│   ├── 0 HDF5_print.py                # HDF5 파일 내부 구조 확인
│   ├── 0 TxtToHdf5.py                 # TXT → HDF5 변환 (단순)
│   ├── 0 convert_txt_to_matrix_hdf5.py # TXT → 2D Matrix HDF5 변환
│   ├── 1 HDF5_Visualize.py            # 단일 질량 이미지 시각화
│   ├── 1 visualize_sims_images.py     # 전체 질량 이미지 배치 시각화
│   ├── 2 HMR_visualize_with_zoom.py   # 고해상도 이미지 줌인 시각화
│   ├── 2 LMR_visualize_with_zoom.py   # 저해상도 이미지 중앙 크롭 시각화
│   ├── 3 Bicubic Interpolation.py     # Bicubic 보간법 업스케일링
│   ├── 4 Alignment check.py           # LR-HR 이미지 정렬 확인 (Overlay)
│   ├── 4-1 rename.py                  # HDF5 데이터셋 이름 수정 도구
│   ├── 4-2 SIFT alignment.py          # SIFT 기반 자동 이미지 정렬
│   ├── 4-3 Manual Registration.py     # 수동 포인트 클릭 기반 이미지 정렬
│   ├── 4-4 Align_Check.py             # 정렬 품질 평가 (NCC, Checkerboard)
│   ├── 5 SRCNN.py                     # SRCNN 모델 학습
│   ├── 5 CNN_Validation.py            # SRCNN 추론 및 결과 시각화
│   ├── 6 SRGAN.py                     # SRGAN 모델 학습
│   ├── 6 GAN_Validation.py            # SRGAN 추론 및 결과 시각화
│   ├── 7 Regression Model.ipynb       # 회귀 모델 실험 노트북
│   ├── 8 Binning Model.ipynb          # 비닝 모델 실험 노트북
│   └── Downsampling_Binning.ipynb     # 다운샘플링 & 비닝 실험 노트북
├── dataset/                            # 원본 데이터 디렉토리 (gitignore)
├── .gitignore
└── README.md
```

---

## 🚀 파이프라인

전체 워크플로우는 **6단계**로 구성됩니다.

### Step 0 — 데이터 변환
ToF-SIMS 장비에서 출력된 `.txt` 파일을 **HDF5** 포맷으로 변환합니다.
- 각 질량(mass) 채널을 2D Matrix로 파싱
- 헤더에서 이미지 크기를 자동 추출
- gzip 압축으로 저장 용량 최적화

### Step 1 — 시각화
HDF5로 변환된 ToF-SIMS 이미지를 확인합니다.
- 단일 질량 이미지 시각화 (`inferno` colormap)
- 전체 질량 이미지 배치 출력 (자동 페이징)

### Step 2 — 줌인 비교
고해상도(HMR)와 저해상도(LMR) 이미지의 디테일을 비교합니다.
- 전체 뷰 + 확대(Zoom-in) 뷰 동시 출력
- ROI(Region of Interest) 확대 및 중앙 크롭

### Step 3 — Bicubic 보간
저해상도 이미지를 Bicubic Interpolation으로 업스케일합니다.
- 딥러닝 모델의 **입력(Input)** 데이터를 생성하는 단계
- OpenCV `cv2.INTER_CUBIC` 사용

### Step 4 — 이미지 정렬 (Registration)
LR과 HR 이미지의 공간적 정렬을 수행합니다.
- **자동 정렬**: SIFT 특징점 매칭 + Homography
- **수동 정렬**: 4-포인트 클릭 기반 원근 변환 (줌인 지원)
- **정렬 검증**: NCC(상관계수), 체커보드, 컬러 오버레이

### Step 5 — SRCNN (Super Resolution CNN)
3-Layer CNN을 이용한 초해상도 복원 모델입니다.
- 아키텍처: `Conv(9×9, 64) → Conv(5×5, 32) → Conv(5×5, 1)`
- Random Patch Sampling (64×64)으로 메모리 효율적 학습
- MSE Loss, Adam Optimizer
- Apple Silicon MPS GPU 가속 지원

### Step 6 — SRGAN (Super Resolution GAN)
GAN 기반 초해상도 복원 모델로, SRCNN보다 **텍스처 디테일**이 뛰어납니다.
- **Generator**: ResNet 구조 (5 Residual Blocks)
- **Discriminator**: 4-Layer CNN with Adaptive Pooling
- Adversarial Loss + L1 Content Loss
- Tile 기반 대용량 이미지 추론 (OOM 방지)

---

## 🛠️ 기술 스택

| 분류 | 라이브러리 |
|------|-----------|
| 딥러닝 | PyTorch |
| 영상 처리 | OpenCV |
| 데이터 포맷 | HDF5 (h5py) |
| 시각화 | Matplotlib |
| 수치 연산 | NumPy, Pandas |

---

## ⚙️ 실행 환경

```bash
# Python 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate

# 의존성 설치
pip install torch torchvision numpy pandas h5py opencv-python matplotlib
```

> **Note**: macOS Apple Silicon 환경에서 `torch.device("mps")`를 통해 GPU 가속을 지원합니다.

---

## 📊 데이터

- **Low Resolution**: `LowMassResolution.hdf5` — 528×528 픽셀
- **High Resolution**: `HighMassResolution.hdf5` — 5166×5166 픽셀
- 각 HDF5 파일에는 다수의 질량(mass) 채널이 2D Matrix로 저장되어 있습니다.

> 데이터 파일(`.hdf5`, `.pth`, `.npy` 등)은 `.gitignore`에 의해 버전 관리에서 제외됩니다.

---

## 📝 License

This project is for research purposes.
