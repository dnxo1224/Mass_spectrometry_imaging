import h5py
import cv2
import matplotlib.pyplot as plt
import numpy as np


def upscale_bicubic(file_path, dataset_name, target_size=(5166, 5166)):
    """
    LR 데이터를 읽어서 Bicubic 방식으로 HR 크기로 뻥튀기(Upscale) 합니다.
    """
    with h5py.File(file_path, 'r') as f:
        # 1. Low Resolution 데이터 로드 (528 x 528)
        lr_data = f[dataset_name][:]

        print(f"Original Shape: {lr_data.shape}")

        # 2. Bicubic Interpolation 적용
        # cv2.resize(원본, (가로, 세로), 보간법 옵션)
        # cv2.INTER_CUBIC: 바이큐빅 보간법 (속도는 느리지만 퀄리티가 좋음)
        # cv2.INTER_LINEAR: 쌍선형 보간법 (기본값, 빠름)
        upscaled_data = cv2.resize(lr_data, dsize=target_size, interpolation=cv2.INTER_CUBIC)

        print(f"Upscaled Shape: {upscaled_data.shape}")

        return lr_data, upscaled_data


# --- 실행 및 비교 ---
file_path = 'LowMassResolution.hdf5'  # LR 데이터가 있는 파일
dataset_name = ' (15) - 16.00 u u'  # 테스트할 질량

lr_img, bicubic_img = upscale_bicubic(file_path, dataset_name, target_size=(5166, 5166))



# --- 시각화: 얼마나 부드러워졌는지 확인 ---
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# 1. 원본 (Low Res) - 픽셀이 튐
ax[0].imshow(lr_img, cmap='inferno', origin='upper')
ax[0].set_title(f"Original LR ({lr_img.shape[0]}x{lr_img.shape[1]})")

# 2. Bicubic 결과 - 부드러워짐 (하지만 디테일은 흐릿함)
ax[1].imshow(bicubic_img, cmap='inferno', origin='upper')
ax[1].set_title(f"Bicubic Upscaled ({bicubic_img.shape[0]}x{bicubic_img.shape[1]})")

plt.show()
