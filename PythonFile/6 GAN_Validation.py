import torch
import torch.nn as nn
import numpy as np
import cv2
import h5py
import matplotlib.pyplot as plt
import math


# ==========================================
# 1. 모델 설계도 (변경 없음)
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv2(self.prelu(self.bn1(self.conv1(x))))
        return x + self.bn2(residual)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(5)])
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        self.conv_out = nn.Conv2d(64, 1, kernel_size=9, padding=4)

    def forward(self, x):
        block1 = self.block1(x)
        res = self.res_blocks(block1)
        block2 = self.block2(res)
        return torch.tanh(self.conv_out(block1 + block2))


# ==========================================
# 2. [핵심] 대용량 이미지 분할 처리 함수
# ==========================================
def process_large_image(model, img_tensor, tile_size=1024, device='cpu'):
    """
    큰 이미지를 tile_size 크기로 잘라서 모델에 넣고 다시 합칩니다.
    메모리 부족(OOM) 방지용 핵심 함수입니다.
    """
    _, _, h, w = img_tensor.shape
    output_img = torch.zeros((1, 1, h, w), device='cpu')  # 결과 담을 빈 캔버스 (CPU)

    # 1. 타일 개수 계산 (올림 처리)
    num_y = math.ceil(h / tile_size)
    num_x = math.ceil(w / tile_size)

    print(f"이미지가 너무 커서 {num_y}x{num_x} = {num_y * num_x}개 타일로 나누어 처리합니다...")

    model.eval()

    with torch.no_grad():
        for i in range(num_y):
            for j in range(num_x):
                # 2. 현재 타일의 좌표 계산
                y_start = i * tile_size
                x_start = j * tile_size
                y_end = min(y_start + tile_size, h)
                x_end = min(x_start + tile_size, w)

                # 3. 타일 잘라내기 (Crop)
                input_tile = img_tensor[:, :, y_start:y_end, x_start:x_end].to(device)

                # 4. 모델 실행 (Inference)
                output_tile = model(input_tile)

                # 5. 결과 이어붙이기 (Stitch)
                # GPU 결과를 CPU로 가져와서 붙임
                output_img[:, :, y_start:y_end, x_start:x_end] = output_tile.cpu()

                # 진행 상황 표시
                print(f"\r  -> 타일 처리 중: {i * num_x + j + 1}/{num_y * num_x}", end="")

    print("\n✅ 전체 이미지 처리 완료!")
    return output_img


# ==========================================
# 3. 결과 확인 메인 함수
# ==========================================
def check_gan_performance(lr_path, hr_path, matrix_path, model_path, target_mass):
    # 장치 설정
    device = torch.device(
        'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")

    # 모델 로드
    model = Generator().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return

    # 데이터 로드
    H = np.load(matrix_path)

    with h5py.File(lr_path, 'r') as f_lr, h5py.File(hr_path, 'r') as f_hr:
        lr_name = target_mass[0]
        hr_name = target_mass[1]

        lr_raw = f_lr[lr_name][:]
        hr_raw = f_hr[hr_name][:]
        h, w = hr_raw.shape

        print("이미지 전처리 중...")

        # Bicubic Upscale & Warp
        lr_upscaled = cv2.resize(lr_raw, (w, h), interpolation=cv2.INTER_CUBIC)
        lr_aligned = cv2.warpPerspective(lr_upscaled, H, (w, h))

        # GAN Normalization (-1 ~ 1)
        def normalize_gan(d):
            vmax = np.percentile(d, 99.5)
            if vmax == 0: vmax = d.max()
            d = np.clip(d, 0, vmax)
            return (d / vmax * 2.0 - 1.0).astype(np.float32)

        input_data = normalize_gan(lr_aligned)

        # Tensor 변환 (Note: input_tensor는 CPU에 둠)
        input_tensor = torch.from_numpy(input_data).unsqueeze(0).unsqueeze(0)

    # --- [수정된 부분] 타일링 처리 호출 ---
    # tile_size=1024: 메모리 상황에 따라 512로 줄이거나 2048로 늘려도 됨
    output_tensor = process_large_image(model, input_tensor, tile_size=1024, device=device)

    # ----------------------------------

    # 후처리 (Denormalization)
    def denorm(img):
        return (img + 1) / 2.0

    img_input = denorm(input_data)
    img_output = denorm(output_tensor.squeeze().cpu().numpy())

    vmax_hr = np.percentile(hr_raw, 99.5)
    if vmax_hr == 0: vmax_hr = hr_raw.max()
    img_target = np.clip(hr_raw / vmax_hr, 0, 1)

    # 시각화
    print("결과 시각화 중...")
    plt.figure(figsize=(15, 10))

    # 상단: 전체 뷰
    plt.subplot(2, 3, 1);
    plt.imshow(img_input, cmap='inferno', vmin=0, vmax=1);
    plt.title("Input (LR Upscaled)");
    plt.axis('off')
    plt.subplot(2, 3, 2);
    plt.imshow(img_output, cmap='inferno', vmin=0, vmax=1);
    plt.title("GAN Result");
    plt.axis('off')
    plt.subplot(2, 3, 3);
    plt.imshow(img_target, cmap='inferno', vmin=0, vmax=1);
    plt.title("Target (Real HR)");
    plt.axis('off')

    # 하단: 줌인 뷰 (중앙 확대)
    cy, cx = h // 2, w // 2
    size = 100

    # 범위 예외 처리
    y1, y2 = max(0, cy - size), min(h, cy + size)
    x1, x2 = max(0, cx - size), min(w, cx + size)

    plt.subplot(2, 3, 4);
    plt.imshow(img_input[y1:y2, x1:x2], cmap='inferno', vmin=0, vmax=1);
    plt.title("Zoom: LR");
    plt.axis('off')
    plt.subplot(2, 3, 5);
    plt.imshow(img_output[y1:y2, x1:x2], cmap='inferno', vmin=0, vmax=1);
    plt.title("Zoom: GAN");
    plt.axis('off')
    plt.subplot(2, 3, 6);
    plt.imshow(img_target[y1:y2, x1:x2], cmap='inferno', vmin=0, vmax=1);
    plt.title("Zoom: HR");
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# ==========================================
# 실행 설정
# ==========================================
LR_FILE = 'LowMassResolution.hdf5'
HR_FILE = 'HighMassResolution.hdf5'
MATRIX_FILE = 'alignment_matrix.npy'
MODEL_PATH = 'srgan_generator_epoch_100.pth'  # 파일 이름 확인 필수!
TARGET = ('20251118001 (19) - Cl- u', '20251118005 (22) - Cl- u')

check_gan_performance(LR_FILE, HR_FILE, MATRIX_FILE, MODEL_PATH, TARGET)