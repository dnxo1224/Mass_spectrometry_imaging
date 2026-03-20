import h5py
import matplotlib.pyplot as plt
import numpy as np


def visualize_low_res_center(file_path, dataset_name, crop_size=500):
    """
    Low Resolution 데이터(528x528)에서 중앙의 crop_size(500) 만큼만 잘라내어 시각화합니다.
    """
    with h5py.File(file_path, 'r') as f:
        if dataset_name not in f:
            print(f"데이터셋 {dataset_name} 을 찾을 수 없습니다.")
            return

        # 1. 데이터 로드
        data = f[dataset_name][:]
        height, width = data.shape

        print(f"원본 데이터 크기: {width} x {height}")

        # 2. 중앙 좌표 및 잘라낼 범위 계산
        # 정중앙 좌표
        cx, cy = width // 2, height // 2

        # 잘라낼 길이의 절반 (500 / 2 = 250)
        half_size = crop_size // 2

        # 슬라이싱 범위 (이미지 범위를 벗어나지 않도록 min/max 처리)
        x_start = max(0, cx - half_size)
        x_end = min(width, cx + half_size)
        y_start = max(0, cy - half_size)
        y_end = min(height, cy + half_size)

        # 3. 데이터 자르기 (Cropping)
        cropped_data = data[y_start:y_end, x_start:x_end]

        print(f"Cropped 크기: {cropped_data.shape[1]} x {cropped_data.shape[0]}")

        # 4. 시각화
        plt.figure(figsize=(8, 8))

        # 명암비 조절 (상위 99.5% 밝기 기준)
        vmax_val = np.percentile(cropped_data, 99.5)
        if vmax_val == 0: vmax_val = cropped_data.max()

        plt.imshow(cropped_data, cmap='inferno', vmax=vmax_val, origin='upper')
        plt.colorbar(label='Intensity')
        plt.title(f"Low Res Center Crop ({crop_size} x {crop_size})")
        plt.xlabel("X Pixel")
        plt.ylabel("Y Pixel")
        plt.show()


# --- 실행 ---
# Low Resolution 파일 경로 (기존에 만든 통합 파일)
low_res_file = 'LowMassResolution.hdf5'

# 실행: 528 크기 중 500을 자르므로 테두리 일부만 잘려 나간 형태가 됩니다.
visualize_low_res_center(low_res_file, ' (15) - 16.00 u u', crop_size=500)