import pandas as pd
import h5py
import os


def text_to_hdf5(txt_folder_path, hdf5_file_name):
    # 1. 새 HDF5 파일 생성
    with h5py.File(hdf5_file_name, 'w') as h5f:

        # 폴더 내의 모든 텍스트 파일 탐색
        for file_name in os.listdir(txt_folder_path):
            if file_name.endswith(".txt"):
                file_path = os.path.join(txt_folder_path, file_name)

                # 파일명에서 데이터셋 이름 추출 (예: mass_12.01.txt -> mass_12.01 u)
                dataset_name = file_name.replace('.txt', ' u')

                print(f"변환 중: {file_name} -> {dataset_name}")

                # 2. 텍스트 파일 읽기 (구분자가 공백인 경우 sep='\s+')
                # 데이터 구조: x, y, intensity
                df = pd.read_csv(file_path, sep='\t', header=None)

                # 3. HDF5 데이터셋으로 저장
                # compression='gzip'을 사용하면 용량을 크게 줄일 수 있습니다.
                h5f.create_dataset(dataset_name, data=df.values, compression='gzip')

    print(f"완료! {hdf5_file_name} 파일이 생성되었습니다.")

# 사용 예시
text_to_hdf5('../dataset/20251118001', 'LowMassResolution.hdf5')