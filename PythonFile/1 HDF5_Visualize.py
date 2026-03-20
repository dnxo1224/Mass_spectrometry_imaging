import h5py
import matplotlib.pyplot as plt


def visualize_tof_sims_matrix(file_path, dataset_name):
    try:
        # 1. 외장 하드 등 경로에서 HDF5 파일 열기
        with h5py.File(file_path, 'r') as f:
            # 2. 데이터셋 가져오기 (스크린샷의 mass_13.02 u 등)
            # f[dataset_name] 자체가 이미 (2100, 875) 또는 (875, 2100) 행렬입니다.
            img_data = f[dataset_name][:]

            print(f"데이터 로드 완료! Shape: {img_data.shape}")

            # 3. 시각화
            plt.figure(figsize=(10, 8))
            # 데이터의 가로세로 비율이 크므로 aspect='auto'를 주면 보기 편합니다.
            im = plt.imshow(img_data, cmap='hot', aspect='auto', origin='lower')

            plt.colorbar(im, label='Intensity')
            plt.title(f"ToF-SIMS Image: {dataset_name}")
            plt.xlabel("X Pixel")
            plt.ylabel("Y Pixel")
            plt.show()

    except Exception as e:
        print(f"에러 발생: {e}")

# 사용 예시 (외장하드 경로 확인)
visualize_tof_sims_matrix('../dataset/Kidney hdf5 data/22020707007_1 kidney_denoised.hdf5', 'mass_13.02 u')