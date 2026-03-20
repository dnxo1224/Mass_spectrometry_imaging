import os
import glob
import re
import pandas as pd
import numpy as np
import h5py


def batch_convert_txt_to_hdf5(input_folder_path, output_hdf5_path):
    """
    폴더 내의 모든 txt 파일을 읽어 하나의 HDF5 파일에 각각의 Matrix로 저장합니다.
    """

    # 1. HDF5 파일 생성 (쓰기 모드 'w')
    with h5py.File(output_hdf5_path, 'w') as h5f:

        # 지정된 폴더 내의 모든 .txt 파일 목록 가져오기
        file_list = glob.glob(os.path.join(input_folder_path, "*.txt"))
        total_files = len(file_list)

        print(f"총 {total_files} 개의 파일을 발견했습니다. 변환을 시작합니다...")

        for idx, txt_file_path in enumerate(file_list):
            file_name = os.path.basename(txt_file_path)

            # 2. 데이터셋 이름 결정
            # 예: 'mass_12.01.txt' -> 'mass_12.01 u' 로 변환
            # (파일명에 따라 적절히 수정 가능)
            dataset_name = file_name.replace('.txt', ' u')

            print(f"[{idx + 1}/{total_files}] 처리 중: {dataset_name} ...", end=" ")

            try:
                # 3. 헤더에서 이미지 크기 파싱 (Width, Height)
                width, height = 0, 0
                with open(txt_file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.startswith('# Image Size:'):
                            match = re.search(r'(\d+)\s*x\s*(\d+)', line)
                            if match:
                                width = int(match.group(1))
                                height = int(match.group(2))
                            break
                        if not line.startswith('#'):
                            break

                if width == 0 or height == 0:
                    print(f"-> [Skip] 헤더에서 이미지 크기를 찾지 못함.")
                    continue

                # 4. 데이터 로드 (주석 무시, 공백 구분)
                # 컬럼: x(0), y(1), intensity(2)
                df = pd.read_csv(txt_file_path, sep='\s+', comment='#', header=None, names=['x', 'y', 'intensity'])

                # 5. 2D Matrix 생성 및 데이터 매핑
                # 빈 행렬 생성 (세로 height, 가로 width)
                matrix_data = np.zeros((height, width), dtype=np.float32)

                # 좌표를 정수형으로 변환하여 인덱싱
                x_idx = df['x'].astype(int)
                y_idx = df['y'].astype(int)

                # 좌표 범위 체크 (데이터가 헤더 사이즈보다 큰 경우 방지)
                valid_mask = (x_idx < width) & (y_idx < height)

                # 유효한 좌표에만 값 채워넣기
                matrix_data[y_idx[valid_mask], x_idx[valid_mask]] = df.loc[valid_mask, 'intensity']

                # 6. HDF5 데이터셋으로 저장
                # 스크린샷처럼 그룹 없이 바로 루트에 저장하거나, 필요하면 그룹 생성 가능
                h5f.create_dataset(dataset_name, data=matrix_data, compression='gzip')
                print("-> 완료")

            except Exception as e:
                print(f"-> [Error] {e}")

    print("\n모든 변환 작업이 종료되었습니다.")
    print(f"저장 위치: {output_hdf5_path}")


# --- 실행 설정 ---
# 데이터셋 폴더 경로 (상대 경로 혹은 절대 경로 입력)
input_folder = '../dataset/20251118001'

# 결과물 HDF5 파일 이름
output_file = 'LowMassResolution.hdf5'

# 함수 실행
if os.path.exists(input_folder):
    batch_convert_txt_to_hdf5(input_folder, output_file)
else:
    print(f"폴더를 찾을 수 없습니다: {input_folder}")

# # 1. 외장하드 경로 설정 (스크린샷 기반)
# usb_path = '/Volumes/NO NAME/인턴 자료/20251118005'
# output_hdf5_path = '/Volumes/NO NAME/인턴 자료/HighMassResolution.hdf5'
# # 결과 파일도 외장하드에 저장하는 것이 좋습니다. (내장 용량 절약)

# # 2. 경로가 진짜 맞는지 테스트
# if os.path.exists(usb_path):
#     print(f"✅ 경로 확인 성공: {usb_path}")
#
#     # 파일 개수 미리보기
#     file_count = len([name for name in os.listdir(usb_path) if name.endswith('.txt')])
#     print(f"📂 발견된 텍스트 파일 개수: {file_count}개")
#     print(f"💾 결과 저장 경로: {output_hdf5_path}")
#
#     # 3. 변환 함수 실행 (이전에 만든 함수 호출)
#     # 시간이 오래 걸릴 수 있으니 실행해두고 기다리세요!
#     print("\n--- 대용량 데이터 변환 시작 ---")
#     batch_convert_txt_to_hdf5(usb_path, output_hdf5_path)
#
# else:
#     print(f"❌ 경로를 찾을 수 없습니다: {usb_path}")
#     print("USB가 제대로 연결되었는지, 폴더 이름에 오타가 없는지 확인해주세요.")