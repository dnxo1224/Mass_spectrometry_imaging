import h5py
import matplotlib.pyplot as plt
import numpy as np


def visualize_with_zoom(file_path, dataset_name, zoom_center=None, zoom_size=500):
    """
    전체 이미지와 확대한(Zoom-in) 이미지를 동시에 보여줍니다.

    :param zoom_center: 확대할 중심 좌표 (x, y). None이면 이미지 정중앙을 선택
    :param zoom_size: 확대해서 볼 윈도우 크기 (픽셀 단위)
    """
    with h5py.File(file_path, 'r') as f:
        if dataset_name not in f:
            print(f"데이터셋 {dataset_name} 을 찾을 수 없습니다.")
            return

        # 데이터 로드
        data = f[dataset_name][:]
        height, width = data.shape

        # 1. 확대할 영역(ROI) 계산
        if zoom_center is None:
            cx, cy = width // 2, height // 2
        else:
            cx, cy = zoom_center

        half_size = zoom_size // 2

        # 배열 슬라이싱 범위 계산 (이미지 밖으로 나가지 않게 처리)
        x_start = max(0, cx - half_size)
        x_end = min(width, cx + half_size)
        y_start = max(0, cy - half_size)
        y_end = min(height, cy + half_size)

        # 데이터 잘라내기 (Cropping)
        zoomed_data = data[y_start:y_end, x_start:x_end]

        # 2. 시각화 (좌: 전체 / 우: 확대)
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        # 전체 이미지 (Downsampling 됨)
        # vmax를 99.5%로 잡아야 픽셀 뭉개짐 속에서도 구조가 보임
        vmax_val = np.percentile(data, 99.5)

        im1 = axes[0].imshow(data, cmap='inferno', vmax=vmax_val, origin='upper')
        axes[0].set_title(f"Full View ({width} x {height})")
        # 확대할 위치에 사각형 표시
        rect = plt.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start,
                             linewidth=2, edgecolor='cyan', facecolor='none')
        axes[0].add_patch(rect)

        # 확대 이미지 (Raw Detail)
        im2 = axes[1].imshow(zoomed_data, cmap='inferno', vmax=vmax_val, origin='upper')
        axes[1].set_title(f"Zoomed View (Center: {cx}, {cy})")

        plt.tight_layout()
        plt.show()


# --- 실행 ---
# High Resolution 파일 경로
file_path = 'HighMassResolution.hdf5'

# 1. 이미지 정중앙을 500x500 픽셀로 확대해서 보기
visualize_with_zoom(file_path, '20251118005 (18) - 16.00 u u', zoom_center=None, zoom_size=800)

# 2. 특정 좌표(예: x=2000, y=2000) 지점을 보고 싶다면
# visualize_with_zoom(file_path, 'mass_12.01 u', zoom_center=(2000, 2000), zoom_size=800)