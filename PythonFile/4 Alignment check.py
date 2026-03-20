import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt

def normalize_data(data):
    """데이터를 0~1 사이로 정규화 (시각화 비교를 위함)"""
    # 노이즈가 섞인 HR 데이터를 위해 상위 99%를 Max로 잡음
    vmax = np.percentile(data, 99)
    if vmax == 0: vmax = data.max()
    normalized = np.clip(data / vmax, 0, 1)
    return normalized

def prepare_training_data_check(lr_path, hr_path, lr_dataset_name, hr_dataset_name):
    """
    LR 데이터와 HR 데이터의 이름이 달라도 각각 읽어와서
    Bicubic으로 Upscale하고, 겹쳐서 위치가 맞는지 보여줍니다.
    """
    # 1. 파일 열기
    f_lr = h5py.File(lr_path, 'r')
    f_hr = h5py.File(hr_path, 'r')

    # 데이터셋 존재 여부 개별 확인
    if lr_dataset_name not in f_lr:
        print(f"Error: LR 파일에 '{lr_dataset_name}' 데이터셋이 없습니다.")
        return
    if hr_dataset_name not in f_hr:
        print(f"Error: HR 파일에 '{hr_dataset_name}' 데이터셋이 없습니다.")
        return

    # 2. 데이터 로드 (각각의 이름으로 로드)
    lr_data = f_lr[lr_dataset_name][:]
    hr_data = f_hr[hr_dataset_name][:]

    print(f"Original LR Shape: {lr_data.shape} (Name: {lr_dataset_name})")
    print(f"Original HR Shape: {hr_data.shape} (Name: {hr_dataset_name})")

    # 3. Bicubic Interpolation (핵심 단계)
    # 목표 크기: HR 데이터의 (가로, 세로)
    target_size = (hr_data.shape[1], hr_data.shape[0])  # (width, height)

    # cv2.INTER_CUBIC: 3차 보간법 (부드럽게 확대)
    lr_upscaled = cv2.resize(lr_data, dsize=target_size, interpolation=cv2.INTER_CUBIC)

    print(f"Upscaled LR Shape: {lr_upscaled.shape} (Completed)")

    # 4. 시각적 정렬 확인 (Alignment Check)
    # 두 데이터의 Shot 수가 다르므로 정규화(Normalize) 후 비교해야 함
    norm_lr = normalize_data(lr_upscaled)
    norm_hr = normalize_data(hr_data)

    plt.figure(figsize=(15, 6))

    # (1) Bicubic 결과 (Input으로 들어갈 데이터)
    plt.subplot(1, 3, 1)
    plt.imshow(norm_lr, cmap='inferno')
    plt.title(f"Input: Bicubic LR\n({lr_dataset_name})")
    plt.axis('off')

    # (2) Real High Res (정답지로 쓸 데이터)
    plt.subplot(1, 3, 2)
    plt.imshow(norm_hr, cmap='inferno')
    plt.title(f"Target: Real HR\n({hr_dataset_name})")
    plt.axis('off')

    # (3) 겹쳐 보기 (Overlay)
    # LR을 녹색(Green), HR을 자홍색(Magenta)으로 합성
    # 겹치는 부분이 회색/흰색이면 정렬 잘 됨. 색이 갈라지면 위치 안 맞음.
    overlay = np.zeros((*target_size, 3))
    overlay[..., 1] = norm_lr  # Green Channel
    overlay[..., 0] = norm_hr  # Red Channel
    overlay[..., 2] = norm_hr  # Blue Channel (R+B = Magenta)

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Overlay Check (Green=LR, Magenta=HR)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # 리소스 정리
    f_lr.close()
    f_hr.close()


# --- 실행 설정 ---
lr_file = 'LowMassResolution.hdf5'
hr_file = 'HighMassResolution.hdf5'

# 여기에 각각의 파일 내부 이름을 정확히 적어주세요.
# 예: LR 파일에는 'mass_12.01 u', HR 파일에는 그냥 '12.01'로 되어 있다면 아래처럼 입력
lr_target_name = '20251118001 (15) - 16.00 u u'   # LR 파일 내부 이름
hr_target_name = '20251118005 (18) - 16.00 u u'          # HR 파일 내부 이름 (예시)

prepare_training_data_check(lr_file, hr_file, lr_target_name, hr_target_name)