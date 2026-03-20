import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import cv2
import numpy as np
import os


# --- 1. SRCNN 모델 정의 (간단하지만 강력함) ---
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        # 레이어 1: 특징 추출 (Feature Extraction)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU()

        # 레이어 2: 비선형 매핑 (Non-linear Mapping)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()

        # 레이어 3: 복원 (Reconstruction)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x


# --- 2. ToF-SIMS 데이터셋 클래스 ---
import random


class SimsDataset(Dataset):
    def __init__(self, lr_path, hr_path, matrix_path, mass_pairs, patch_size=64, iterations_per_epoch=1000):
        """
        패치를 미리 만들지 않고, 원본 이미지를 메모리에 들고 있다가
        학습 때마다 랜덤하게 잘라서 줍니다. (대기 시간 0초)
        """
        self.lr_images = []
        self.hr_images = []
        self.patch_size = patch_size
        self.iterations = iterations_per_epoch  # 1 에포크당 몇 번 학습할지 설정

        # 변환 행렬 로드
        H = np.load(matrix_path)

        print("이미지 데이터를 RAM에 로드 중입니다... (패치 생성 안 함)")

        with h5py.File(lr_path, 'r') as f_lr, h5py.File(hr_path, 'r') as f_hr:
            for idx, (lr_name, hr_name) in enumerate(mass_pairs):
                if lr_name not in f_lr or hr_name not in f_hr: continue

                # 데이터 로드
                lr_raw = f_lr[lr_name][:]
                hr_raw = f_hr[hr_name][:]

                h, w = hr_raw.shape

                # 1. LR Upscaling & Alignment (한 번만 수행)
                try:
                    lr_upscaled = cv2.resize(lr_raw, (w, h), interpolation=cv2.INTER_CUBIC)
                    lr_aligned = cv2.warpPerspective(lr_upscaled, H, (w, h))
                except:
                    continue

                # 2. 정규화 (미리 해둠)
                def normalize(d):
                    vmax = np.percentile(d, 99)
                    if vmax == 0: vmax = d.max()
                    return np.clip(d / vmax, 0, 1).astype(np.float32)

                self.lr_images.append(normalize(lr_aligned))
                self.hr_images.append(normalize(hr_raw))

                print(f"\r[{idx + 1}/{len(mass_pairs)}] 이미지 로드 완료", end="")

        print(f"\n✅ 로드 완료! 총 {len(self.lr_images)}쌍의 이미지가 준비되었습니다.")
        print("이제 기다림 없이 바로 학습이 시작됩니다.")

    def __len__(self):
        # 데이터셋의 길이를 임의로 지정 (한 에포크당 반복 횟수)
        return self.iterations

    def __getitem__(self, idx):
        # 1. 랜덤하게 이미지 쌍 하나 선택
        img_idx = random.randint(0, len(self.lr_images) - 1)
        lr_img = self.lr_images[img_idx]
        hr_img = self.hr_images[img_idx]

        h, w = lr_img.shape

        # 2. 랜덤한 위치(x, y) 선택 (Random Crop)
        # 이미지 범위를 벗어나지 않게 좌표 선정
        y = random.randint(0, h - self.patch_size)
        x = random.randint(0, w - self.patch_size)

        # 3. 그 위치만 싹둑 잘라서 리턴
        patch_in = lr_img[y:y + self.patch_size, x:x + self.patch_size]
        patch_tar = hr_img[y:y + self.patch_size, x:x + self.patch_size]

        # 차원 추가 (H, W) -> (1, H, W)
        return torch.from_numpy(patch_in[np.newaxis, ...]), torch.from_numpy(patch_tar[np.newaxis, ...])


# 학습 실행 코드 (Training Loop)
def train_model():
    # --- 설정 ---
    LR_FILE = 'LowMassResolution.hdf5'
    HR_FILE = 'HighMassResolution.hdf5'
    MATRIX_FILE = 'alignment_matrix.npy'

    # 학습에 사용할 질량 리스트 (많을수록 좋습니다)
    # 예시로 탄소(C), CH2, CH3 등을 짝지어 줍니다.
    # (LR 이름, HR 이름)
    # 학습에 사용할 (LR 데이터셋 이름, HR 데이터셋 이름) 쌍
    MASS_PAIRS = [
        # --- Total Ion ---
        ('20251118001 (0) - total u', '20251118005 (0) - total u'),

        # --- Low Mass Range ---
        ('20251118001 (10) - 196.96 u u', '20251118005 (13) - 196.96 u u'),
        #('20251118001 (11) - 394.93 u u', '20251118005 (14) - 394.93 u u'),
        #('20251118001 (12) - 27.98 u u', '20251118005 (15) - 27.98 u u'),
        #('20251118001 (13) - 43.97 u u', '20251118005 (16) - 43.97 u u'),
        #('20251118001 (14) - 59.97 u u', '20251118005 (17) - 59.97 u u'),
        #('20251118001 (15) - 16.00 u u', '20251118005 (18) - 16.00 u u'),

        # --- Chemical Formulas ---
        # ('20251118001 (16) - OH- u', '20251118005 (19) - OH- u'),
        #('20251118001 (17) - C_2H- u', '20251118005 (20) - C_2H- u'),
        ('20251118001 (18) - CN- u', '20251118005 (21) - CN- u'),
        ('20251118001 (19) - Cl- u', '20251118005 (22) - Cl- u'),
        ('20251118001 (20) - ^37Cl- u', '20251118005 (23) - ^37Cl- u'),
        ('20251118001 (21) - CNO- u', '20251118005 (24) - CNO- u'),
        ('20251118001 (24) - C_2H_3O_2- u', '20251118005 (27) - C_2H_3O_2- u'),
        #('20251118001 (25) - CHSO- u', '20251118005 (28) - CHSO- u'),
        #('20251118001 (26) - SO_2- u', '20251118005 (29) - SO_2- u'),
        ('20251118001 (28) - SO_3- u', '20251118005 (31) - SO_3- u'),
        ('20251118001 (29) - C_2H_2Cl_2- u', '20251118005 (32) - C_2H_2Cl_2- u'),
        ('20251118001 (30) - C_2H_3Cl_2- u', '20251118005 (33) - C_2H_3Cl_2- u'),
        ('20251118001 (31) - Si_3CH_3- u', '20251118005 (34) - Si_3CH_3- u'),
        ('20251118001 (32) - I- u', '20251118005 (35) - I- u'),
        #('20251118001 (33) - Si_2CH_5S_2- u', '20251118005 (36) - Si_2CH_5S_2- u'),
        #('20251118001 (34) - C_3HO_5F_2- u', '20251118005 (37) - C_3HO_5F_2- u'),
        #('20251118001 (37) - InO_3H- u', '20251118005 (40) - InO_3H- u'),
        #('20251118001 (38) - InH_2O_3- u', '20251118005 (41) - InH_2O_3- u'),

        # --- High Mass Range ---
        #('20251118001 (42) - 154.89 u u', '20251118005 (45) - 154.89 u u'),
        #('20251118001 (43) - C_6H_5PO_4- u', '20251118005 (46) - C_6H_5PO_4- u'),
        ('20251118001 (50) - 248.95 u u', '20251118005 (52) - 248.95 u u'),
        ('20251118001 (51) - 257.94 u u', '20251118005 (53) - 257.94 u u'),
        ('20251118001 (52) - 266.91 u u', '20251118005 (54) - 266.91 u u'),
        ('20251118001 (53) - 268.90 u u', '20251118005 (55) - 268.90 u u'),
        #('20251118001 (54) - 270.90 u u', '20251118005 (56) - 270.90 u u'),
        #('20251118001 (55) - 273.93 u u', '20251118005 (57) - 273.93 u u'),
        #('20251118001 (56) - 328.90 u u', '20251118005 (58) - 328.90 u u'),
        #('20251118001 (57) - 330.90 u u', '20251118005 (59) - 330.90 u u'),
        #('20251118001 (58) - 358.84 u u', '20251118005 (60) - 358.84 u u'),
        #('20251118001 (59) - 393.92 u u', '20251118005 (61) - 393.92 u u'),
        #('20251118001 (60) - 390.90 u u', '20251118005 (62) - 390.90 u u'),
        #('20251118001 (61) - 419.93 u u', '20251118005 (63) - 419.93 u u'),
        ('20251118001 (62) - 428.90 u u', '20251118005 (64) - 428.90 u u'),
        #('20251118001 (63) - 498.85 u u', '20251118005 (65) - 498.85 u u'),
        #('20251118001 (64) - 500.85 u u', '20251118005 (66) - 500.85 u u'),
        #('20251118001 (65) - 489.88 u u', '20251118005 (67) - 489.88 u u'),
        #('20251118001 (66) - 480.89 u u', '20251118005 (68) - 480.89 u u'),
        #('20251118001 (67) - 520.83 u u', '20251118005 (69) - 520.83 u u'),
        #('20251118001 (68) - 502.84 u u', '20251118005 (70) - 502.84 u u'),
        #('20251118001 (69) - 590.89 u u', '20251118005 (71) - 590.89 u u'),
        #'20251118001 (70) - 660.83 u u', '20251118005 (72) - 660.83 u u'),
        #('20251118001 (71) - 692.81 u u', '20251118005 (73) - 692.81 u u'),
        #('20251118001 (72) - 752.78 u u', '20251118005 (74) - 752.78 u u'),
    ]

    BATCH_SIZE = 16
    EPOCHS = 50
    LEARNING_RATE = 0.0001

    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")  # 맥북 GPU 가속 사용
    else:
        DEVICE = torch.device("cpu")

    # 1. 데이터셋 및 로더 생성
    dataset = SimsDataset(LR_FILE, HR_FILE, MATRIX_FILE, MASS_PAIRS, patch_size=64)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. 모델 및 최적화 도구 준비
    model = SRCNN().to(DEVICE)
    criterion = nn.MSELoss()  # 손실함수: Mean Squared Error (픽셀 값 차이 최소화)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n--- 학습 시작 ---")
    model.train()

    for epoch in range(EPOCHS):
        epoch_loss = 0.0

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            # Forward (예측)
            outputs = model(inputs)

            # Loss 계산 (정답 HR과 비교)
            loss = criterion(outputs, targets)

            # Backward (학습)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}] Loss: {avg_loss:.6f}")

        # 10 에포크마다 모델 저장
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'srcnn_model_epoch_{epoch + 1}.pth')

    print("학습 완료! 모델이 저장되었습니다.")

    return model  # 학습된 모델 반환


# --- 실행 ---
trained_model = train_model()