import h5py
import matplotlib.pyplot as plt
import numpy as np
import math


def visualize_sims_images_all(hdf5_path, target_masses=None, cols=3, batch_size=6, clim_percentile=98):
    """
    HDF5 파일의 모든 질량 이미지를 batch_size 만큼씩 끊어서 끝까지 시각화합니다.

    :param hdf5_path: HDF5 파일 경로
    :param target_masses: 보고 싶은 질량 이름 리스트 (None일 경우 파일 전체)
    :param cols: 한 줄에 보여줄 이미지 개수
    :param batch_size: 한 번의 Figure에 보여줄 이미지 개수 (기본 6개)
    :param clim_percentile: 명암 조절 백분위수
    """
    with h5py.File(hdf5_path, 'r') as f:
        # 1. 파일 내의 모든 데이터셋 이름 가져오기
        all_keys = sorted(list(f.keys()))

        # 2. 전체 대상 키(Key) 리스트 확정
        if target_masses is None:
            # 전체 다 보기
            display_keys_total = all_keys
        else:
            # 사용자가 요청한 것만 추리기
            display_keys_total = [k for k in target_masses if k in f]
            if not display_keys_total:
                print("요청하신 질량 데이터를 파일에서 찾을 수 없습니다.")
                return

        total_images = len(display_keys_total)
        print(f"총 {total_images} 개의 이미지를 {batch_size}개씩 나누어 출력합니다.")

        # 3. Batch 단위로 루프 (0, 6, 12, 18 ... 순서로 진행)
        for i in range(0, total_images, batch_size):
            # 현재 배치의 키 리스트 슬라이싱 (예: 0~5, 6~11 ...)
            current_batch_keys = display_keys_total[i: i + batch_size]
            current_count = len(current_batch_keys)

            # 행(row) 개수 계산
            rows = math.ceil(current_count / cols)

            # Figure 생성
            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))

            # axes가 1개이거나 1차원 배열일 때 등 처리 일관성 확보
            if rows * cols == 1:
                axes = np.array([axes])
            axes = np.array(axes).flatten()

            print(f"\n--- Batch {i // batch_size + 1} (Index {i} ~ {i + current_count - 1}) ---")

            # 현재 배치 내부 반복
            for j, key in enumerate(current_batch_keys):
                ax = axes[j]
                data = f[key][:]

                # 명암 보정
                vmax_val = np.percentile(data, clim_percentile)
                if vmax_val == 0: vmax_val = data.max()

                # 이미지 출력
                im = ax.imshow(data, cmap='inferno', vmax=vmax_val, origin='upper', interpolation='nearest')
                ax.set_title(f"{key}", fontsize=12, fontweight='bold')
                ax.axis('off')

            # 남은 빈 서브플롯 끄기 (마지막 배치에서 빈칸이 생길 때)
            for k in range(current_count, len(axes)):
                axes[k].axis('off')

            plt.tight_layout()
            plt.show()
            # 일반 파이썬 실행환경(pycharm 등)에서는 창을 닫아야 다음 배치가 뜹니다.
            # 주피터 노트북에서는 아래로 쭉 나열됩니다.


# --- 실행 부분 ---
file_path = 'HighMassResolution.hdf5'

# 실행: 전체 데이터를 6개씩 끊어서 끝까지 보여줍니다.
visualize_sims_images_all(file_path, target_masses=None, cols=3, batch_size=6)