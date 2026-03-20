import h5py


def print_root_keys(file_path):
    print(f"--- 파일: {file_path} 의 최상위 목록 ---")

    with h5py.File(file_path, 'r') as f:
        # f.keys()는 최상위 그룹/데이터셋의 이름만 반환합니다.
        keys = list(f.keys())
        for key in keys:
            print(key)

        print(f"\n총 {len(keys)} 개의 항목이 있습니다.")


# 사용 예시
print_root_keys('HighMassResolution.hdf5')