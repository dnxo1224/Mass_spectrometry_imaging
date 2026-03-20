import torch
import torch.nn as nn
import numpy as np
import cv2
import h5py
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset


# ==========================================
# 1. 필요한 클래스 다시 정의 (설계도)
# ==========================================

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x


class SimsDataset(Dataset):
    def __init__(self, lr_path, hr_path, matrix_path, mass_pairs, patch_size=64, iterations_per_epoch=100):
        self.lr_images = []
        self.hr_images = []
        self.patch_size = patch_size
        self.iterations = iterations_per_epoch

        # 변환 행렬 로드
        H = np.load(matrix_path)

        print("데이터 로드 중... (잠시만 기다리세요)")
        with h5py.File(lr_path, 'r') as f_lr, h5py.File(hr_path, 'r') as f_hr:
            for idx, (lr_name, hr_name) in enumerate(mass_pairs):
                if lr_name not in f_lr or hr_name not in f_hr: continue

                lr_raw = f_lr[lr_name][:]
                hr_raw = f_hr[hr_name][:]
                h, w = hr_raw.shape

                # Upscale & Align
                try:
                    lr_upscaled = cv2.resize(lr_raw, (w, h), interpolation=cv2.INTER_CUBIC)
                    lr_aligned = cv2.warpPerspective(lr_upscaled, H, (w, h))
                except:
                    continue

                # Normalize
                def normalize(d):
                    vmax = np.percentile(d, 99)
                    if vmax == 0: vmax = d.max()
                    return np.clip(d / vmax, 0, 1).astype(np.float32)

                self.lr_images.append(normalize(lr_aligned))
                self.hr_images.append(normalize(hr_raw))
        print("✅ 데이터 준비 완료!")

    def __len__(self):
        return self.iterations

    def __getitem__(self, idx):
        img_idx = random.randint(0, len(self.lr_images) - 1)
        lr_img = self.lr_images[img_idx]
        hr_img = self.hr_images[img_idx]

        h, w = lr_img.shape
        y = random.randint(0, h - self.patch_size)
        x = random.randint(0, w - self.patch_size)

        patch_in = lr_img[y:y + self.patch_size, x:x + self.patch_size]
        patch_tar = hr_img[y:y + self.patch_size, x:x + self.patch_size]

        return torch.from_numpy(patch_in[np.newaxis, ...]), torch.from_numpy(patch_tar[np.newaxis, ...])


# ==========================================
# 2. 결과 시각화 함수
# ==========================================
def visualize_result(model, dataset, device):
    model.eval()

    # 데이터셋에서 랜덤하게 하나 뽑기
    lr_tensor, hr_tensor = dataset[0]
    input_tensor = lr_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output_tensor = model(input_tensor)

    img_lr = lr_tensor.squeeze().cpu().numpy()
    img_hr = hr_tensor.squeeze().cpu().numpy()
    img_output = output_tensor.squeeze().cpu().numpy()

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img_lr, cmap='inferno', vmin=0, vmax=1)
    plt.title("Input (Low Res Upscaled)")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(img_output, cmap='inferno', vmin=0, vmax=1)
    plt.title("Result (SRCNN Output)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(img_hr, cmap='inferno', vmin=0, vmax=1)
    plt.title("Target (Real High Res)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# ==========================================
# 3. 메인 실행 부분 (여기를 수정하세요)
# ==========================================

# 1) 파일 경로 설정
LR_FILE = 'LowMassResolution.hdf5'
HR_FILE = 'HighMassResolution.hdf5'
MATRIX_FILE = 'alignment_matrix.npy'
MODEL_PATH = 'srcnn_model_epoch_50.pth'  # <-- [중요] 저장된 모델 파일 이름 확인!

# 테스트용 질량 쌍 (하나만 있어도 됨)
MASS_PAIRS = [('20251118001 (15) - 16.00 u u', '20251118005 (18) - 16.00 u u')]

# 2) 장치 설정
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print(f"Using Device: {DEVICE}")

# 3) 모델 불러오기 (Load Model)
model = SRCNN().to(DEVICE)
try:
    # 저장된 가중치(weights)를 모델에 덮어씌움
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("✅ 모델을 성공적으로 불러왔습니다.")
except FileNotFoundError:
    print(f"❌ 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
    print("학습 코드에서 저장한 .pth 파일 이름이 맞는지 확인해주세요.")
    exit()

# 4) 데이터셋 준비 (Load Dataset)
dataset = SimsDataset(LR_FILE, HR_FILE, MATRIX_FILE, MASS_PAIRS)

# 5) 시각화 실행
visualize_result(model, dataset, DEVICE)