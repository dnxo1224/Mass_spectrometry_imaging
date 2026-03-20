import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

# 맥북 한글 폰트 설정
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False


def evaluate_alignment(lr_path, hr_path, matrix_path, target_mass):
    # 1. 데이터 및 행렬 로드
    H = np.load(matrix_path)

    with h5py.File(lr_path, 'r') as f_lr, h5py.File(hr_path, 'r') as f_hr:
        lr_name, hr_name = target_mass
        lr_raw = f_lr[lr_name][:]
        hr_raw = f_hr[hr_name][:]

    h, w = hr_raw.shape

    # 2. LR 이미지 변환 (Upscale + Warp)
    # 수동 정렬 때와 똑같이 Bicubic 확대 후 행렬 적용
    lr_upscaled = cv2.resize(lr_raw, (w, h), interpolation=cv2.INTER_CUBIC)
    lr_aligned = cv2.warpPerspective(lr_upscaled, H, (w, h))

    # 3. 정규화 (0~1 범위로 맞춤)
    # 비교를 위해 밝기를 비슷하게 맞춰줍니다.
    def normalize(d):
        vmax = np.percentile(d, 99)
        if vmax == 0: vmax = d.max()
        return np.clip(d / vmax, 0, 1)

    img_lr = normalize(lr_aligned)
    img_hr = normalize(hr_raw)

    # -------------------------------------------------------
    # [평가 1] 정량적 수치 계산 (NCC: 상관계수)
    # -------------------------------------------------------
    # 1차원 배열로 펼쳐서 상관계수 계산
    score = np.corrcoef(img_lr.flatten(), img_hr.flatten())[0, 1]
    print(f"\n📊 정렬 점수 (Correlation): {score:.4f}")
    print("   (1.0에 가까울수록 완벽하게 일치함)")

    if score < 0.5:
        print("⚠️ 경고: 점수가 낮습니다. 정렬이 잘못되었거나 데이터 특성이 많이 다릅니다.")

    # -------------------------------------------------------
    # [평가 2] 체커보드 시각화 (Checkerboard)
    # -------------------------------------------------------
    def create_checkerboard(img1, img2, tile_size=200):
        rows, cols = img1.shape
        checkerboard = np.zeros_like(img1)

        for y in range(0, rows, tile_size):
            for x in range(0, cols, tile_size):
                # 행/열 인덱스 합이 짝수면 img1, 홀수면 img2 사용
                if ((y // tile_size) + (x // tile_size)) % 2 == 0:
                    checkerboard[y:y + tile_size, x:x + tile_size] = img1[y:y + tile_size, x:x + tile_size]
                else:
                    checkerboard[y:y + tile_size, x:x + tile_size] = img2[y:y + tile_size, x:x + tile_size]

        return checkerboard

    # 타일 크기 200픽셀로 설정 (화면 크기에 따라 조절 가능)
    checker_img = create_checkerboard(img_lr, img_hr, tile_size=200)

    # -------------------------------------------------------
    # [평가 3] 오버레이 (Overlay)
    # -------------------------------------------------------
    overlay = np.zeros((h, w, 3))
    overlay[..., 0] = img_hr  # Red 채널: HR
    overlay[..., 1] = img_lr  # Green 채널: LR
    # 겹치는 부분은 노란색(R+G)으로 보임.
    # 빨간색이나 초록색이 따로 놀면 정렬이 안 된 것.

    # -------------------------------------------------------
    # 결과 출력
    # -------------------------------------------------------
    plt.figure(figsize=(18, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(checker_img, cmap='inferno')
    plt.title(f"체커보드 비교 (Tile Size=200)\n경계선이 끊어지지 않아야 함")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title("컬러 오버레이 (Red: HR / Green: LR)\n잘 맞으면 전체적으로 노란색 톤")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# --- 실행 설정 ---
LR_FILE = 'LowMassResolution.hdf5'
HR_FILE = 'HighMassResolution.hdf5'
MATRIX_FILE = 'alignment_matrix.npy'

# 비교할 질량 (모서리가 잘 보이는 질량을 선택하는 게 좋습니다)
TARGET = ('20251118001 (19) - Cl- u', '20251118005 (22) - Cl- u')
evaluate_alignment(LR_FILE, HR_FILE, MATRIX_FILE, TARGET)