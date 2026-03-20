import h5py


def manual_rename_tool(file_path):
    """
    사용자가 직접 입력을 통해 HDF5 내부 이름을 하나씩 수정하는 도구
    """
    print(f"파일을 엽니다: {file_path}")

    # 'r+' 모드로 파일 열기 (수정 가능)
    with h5py.File(file_path, 'r+') as f:
        while True:
            # 1. 현재 파일에 있는 리스트 보여주기 (상위 10개만 예시로)
            current_keys = sorted(list(f.keys()))
            print("\n--- 현재 데이터셋 목록 (일부) ---")
            print(current_keys[:10])
            if len(current_keys) > 10: print("...")

            print("\n[수정 모드] '옛날이름, 새이름' 형식으로 입력하세요.")
            print("(종료하려면 'q' 또는 'exit' 입력)")

            # 2. 사용자 입력 받기
            user_input = input("입력 >> ")

            if user_input.lower() in ['q', 'exit', 'quit']:
                print("수정을 종료합니다.")
                break

            # 3. 입력값 파싱 및 이름 변경
            try:
                # 콤마(,)로 구분해서 옛날 이름과 새 이름 나눔
                if ',' not in user_input:
                    print("❌ 형식이 잘못되었습니다. 'mass_12.01, mass_12.01 u' 처럼 콤마로 구분해주세요.")
                    continue

                old_name, new_name = map(str.strip, user_input.split(','))

                if old_name not in f:
                    print(f"❌ '{old_name}'을(를) 찾을 수 없습니다.")
                    continue

                if new_name in f:
                    print(f"❌ '{new_name}'은(는) 이미 존재하는 이름입니다.")
                    continue

                # 진짜 변경 실행
                f.move(old_name, new_name)
                print(f"✅ 변경 완료: {old_name} ---> {new_name}")

            except Exception as e:
                print(f"오류 발생: {e}")


# --- 사용법 ---
# High Resolution 파일 경로를 넣으세요
target_file = 'HighMassResolution.hdf5'

# 도구 실행 (터미널처럼 입력 대기 상태가 됩니다)
manual_rename_tool(target_file)