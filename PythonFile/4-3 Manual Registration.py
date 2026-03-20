import matplotlib

# PyCharm에서 새 창을 띄우기 위한 필수 설정
matplotlib.use('TkAgg')

import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

# 맥북 한글 폰트 설정
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False


def get_points_with_zoom(image, title):
    """
    1. 전체 이미지를 보여주고 대략적인 위치를 클릭받음
    2. 클릭한 주변을 확대해서 보여줌
    3. 정밀한 위치를 다시 클릭받음
    위 과정을 4번(좌상, 우상, 우하, 좌하 순서) 반복함.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # 창을 맨 앞으로 (맥북 버그 방지)
    try:
        plt.get_current_fig_manager().window.attributes('-topmost', 1)
        plt.get_current_fig_manager().window.attributes('-topmost', 0)
    except:
        pass

    final_points = []

    # 순서 가이드 (혼동 방지를 위해 순서를 강제함)
    corner_names = ["1. 좌상단 (Top-Left)", "2. 우상단 (Top-Right)",
                    "3. 우하단 (Bottom-Right)", "4. 좌하단 (Bottom-Left)"]

    # 줌 범위 (반경 몇 픽셀을 보여줄지)
    # 5166x5166 이미지라면 200~300 정도가 적당합니다.
    zoom_radius = 200

    print(f"[{title}] 창이 떴습니다. 순서대로 진행해주세요.")

    for i in range(4):
        # -----------------------------------------------
        # 1단계: 전체 뷰 (Full View)
        # -----------------------------------------------
        ax.clear()
        ax.imshow(image, cmap='inferno', interpolation='none')
        ax.set_title(f"[{title}]\n{corner_names[i]} 근처를 '대충' 클릭하세요.")
        ax.axis('off')
        fig.canvas.draw()

        # 첫 번째 클릭 (대략적 위치)
        rough_point = plt.ginput(n=1, timeout=-1)
        if not rough_point: break  # 중간에 창 끄면 종료

        rx, ry = rough_point[0]

        # -----------------------------------------------
        # 2단계: 줌 뷰 (Zoom View)
        # -----------------------------------------------
        # 이미지 범위를 벗어나지 않도록 좌표 계산
        h, w = image.shape
        x1 = int(max(0, rx - zoom_radius))
        x2 = int(min(w, rx + zoom_radius))
        y1 = int(max(0, ry - zoom_radius))
        y2 = int(min(h, ry + zoom_radius))

        # 확대된 부분 잘라내기
        cropped_img = image[y1:y2, x1:x2]

        ax.clear()
        # interpolation='nearest'를 써야 픽셀이 선명하게 보임
        ax.imshow(cropped_img, cmap='inferno', interpolation='nearest')
        ax.set_title(f"확대됨! {corner_names[i]}의 '정확한' 픽셀을 클릭하세요.")
        ax.axis('off')
        fig.canvas.draw()

        # 두 번째 클릭 (정밀 위치)
        precise_point = plt.ginput(n=1, timeout=-1)
        if not precise_point: break

        px, py = precise_point[0]

        # -----------------------------------------------
        # 좌표 변환 (크롭 이미지 좌표 -> 전체 이미지 좌표)
        # -----------------------------------------------
        global_x = x1 + px
        global_y = y1 + py

        final_points.append([global_x, global_y])
        print(f" -> {corner_names[i]} 선택 완료: ({global_x:.1f}, {global_y:.1f})")

    plt.close(fig)
    return np.float32(final_points)


def align_and_save_matrix(lr_path, hr_path, lr_name, hr_name):
    # 1. 데이터 로드
    with h5py.File(lr_path, 'r') as f_lr, h5py.File(hr_path, 'r') as f_hr:
        lr_data = f_lr[lr_name][:]
        hr_data = f_hr[hr_name][:]

    # 2. LR 이미지 확대
    h_hr, w_hr = hr_data.shape
    lr_upscaled = cv2.resize(lr_data, (w_hr, h_hr), interpolation=cv2.INTER_CUBIC)

    # 시각화용 8-bit 변환
    def to_8bit(img):
        vmax = np.percentile(img, 98)  # 밝기 조절 (95~99 사이 조절)
        if vmax == 0: vmax = img.max()
        return (np.clip(img / vmax, 0, 1) * 255).astype(np.uint8)

    img_lr_view = to_8bit(lr_upscaled)
    img_hr_view = to_8bit(hr_data)

    # 3. 사용자 클릭 (수정된 함수 사용)
    print("\n--- 1단계: Low Resolution (LR) 정밀 클릭 ---")
    pts_src = get_points_with_zoom(img_lr_view, "LR 이미지")
    if len(pts_src) < 4: return None

    print("\n--- 2단계: High Resolution (HR) 정밀 클릭 ---")
    pts_dst = get_points_with_zoom(img_hr_view, "HR 이미지")
    if len(pts_dst) < 4: return None

    # 4. 변환 행렬 계산
    print("\n변환 행렬 계산 중...")
    H, status = cv2.findHomography(pts_src, pts_dst)

    # 5. 저장
    save_name = 'alignment_matrix.npy'
    np.save(save_name, H)
    print(f"\n✅ 저장 완료! 변환 행렬이 '{save_name}' 파일로 저장되었습니다.")

    # 6. 결과 확인
    aligned_lr = cv2.warpPerspective(lr_upscaled, H, (w_hr, h_hr))

    plt.figure(figsize=(10, 10))
    overlay = np.zeros((*hr_data.shape, 3))

    def norm(d):
        return np.clip(d / np.percentile(d, 99), 0, 1)

    overlay[..., 1] = norm(aligned_lr)
    overlay[..., 0] = norm(hr_data)
    overlay[..., 2] = norm(hr_data)

    plt.imshow(overlay)
    plt.title("정렬 결과 확인 (Overlay)")
    plt.axis('off')
    plt.show()


# --- 실행 설정 ---
lr_file = 'LowMassResolution.hdf5'
hr_file = 'HighMassResolution.hdf5'

# 모서리가 가장 뚜렷하게 보이는 이온을 선택하는 것이 중요합니다!
# Cl- (염소)가 보통 테두리가 잘 보입니다.
lr_dataset = '20251118001 (19) - Cl- u'
hr_dataset = '20251118005 (22) - Cl- u'

align_and_save_matrix(lr_file, hr_file, lr_dataset, hr_dataset)