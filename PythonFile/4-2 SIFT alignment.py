import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt


def to_8bit(data):
    """
    SIFT는 0~255 범위의 uint8 이미지에서만 작동하므로 변환이 필수입니다.
    """
    # 1. 노이즈 제거를 위해 상위 1% 값으로 클리핑 (Contrast 향상)
    vmax = np.percentile(data, 99)
    if vmax == 0: vmax = data.max()

    # 2. 0~255로 정규화
    norm_data = np.clip(data / vmax, 0, 1) * 255
    return norm_data.astype(np.uint8)


def align_images_sift(lr_path, hr_path, lr_name, hr_name):
    """
    SIFT 알고리즘을 사용하여 LR 이미지를 HR 이미지 좌표계에 맞춰 변환(Warp)합니다.
    """
    # 1. 데이터 로드
    with h5py.File(lr_path, 'r') as f_lr, h5py.File(hr_path, 'r') as f_hr:
        if lr_name not in f_lr or hr_name not in f_hr:
            print("데이터셋 이름을 확인해주세요.")
            return None, None

        lr_raw = f_lr[lr_name][:]
        hr_raw = f_hr[hr_name][:]

    # 2. LR 이미지를 HR 크기로 1차 확대 (Bicubic)
    # 특징점을 찾으려면 스케일이 비슷해야 유리합니다.
    h_hr, w_hr = hr_raw.shape
    lr_upscaled = cv2.resize(lr_raw, (w_hr, h_hr), interpolation=cv2.INTER_CUBIC)

    # 3. SIFT를 위한 8-bit 변환
    img_lr_8bit = to_8bit(lr_upscaled)
    img_hr_8bit = to_8bit(hr_raw)

    # 4. SIFT 특징점 검출 (Detect SIFT features)
    sift = cv2.SIFT_create()

    # keypoints(특징점 위치), descriptors(특징 벡터) 추출
    kp1, des1 = sift.detectAndCompute(img_lr_8bit, None)  # LR (Query)
    kp2, des2 = sift.detectAndCompute(img_hr_8bit, None)  # HR (Train)

    print(f"특징점 개수 -> LR: {len(kp1)}개, HR: {len(kp2)}개")

    # 5. 특징점 매칭 (FLANN or BFMatcher)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)  # k=2: 가장 가까운 2개 찾기

    # 6. 좋은 매칭점 골라내기 (Lowe's Ratio Test)
    # 두 번째로 가까운 점보다 훨씬 더 가까운 점만 선택 (오매칭 제거)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    print(f"유효한 매칭 점 개수: {len(good_matches)}개")

    if len(good_matches) < 4:
        print("매칭 점이 너무 적어 정렬할 수 없습니다. (최소 4개 필요)")
        return None, None

    # 매칭 결과 시각화 (선 연결)
    img_matches = cv2.drawMatches(img_lr_8bit, kp1, img_hr_8bit, kp2, good_matches[:20], None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(12, 6))
    plt.imshow(img_matches)
    plt.title("SIFT Feature Matching (Top 20)")
    plt.show()

    # 7. 변환 행렬(Homography) 계산
    # 좋은 매칭점들의 좌표를 추출
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # RANSAC 알고리즘으로 이상치(Outlier)를 제외하고 변환 행렬 계산
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 8. LR 이미지를 변환 (Warp Perspective)
    # 계산된 M 행렬을 이용해 LR 이미지를 비틀어서 HR에 맞춤
    aligned_lr = cv2.warpPerspective(lr_upscaled, M, (w_hr, h_hr))

    return aligned_lr, hr_raw


# --- 실행 및 검증 ---
lr_file = 'LowMassResolution.hdf5'
hr_file = 'HighMassResolution.hdf5'
lr_name = '20251118001 (15) - 16.00 u u'
hr_name = '20251118005 (18) - 16.00 u u'  # 본인의 파일 이름에 맞게 수정

aligned_lr, target_hr = align_images_sift(lr_file, hr_file, lr_name, hr_name)

if aligned_lr is not None:
    # 정렬 결과 확인 (Overlay)
    plt.figure(figsize=(10, 10))

    # 시각화용 정규화
    norm_aligned = to_8bit(aligned_lr) / 255.0
    norm_hr = to_8bit(target_hr) / 255.0

    overlay = np.zeros_like(norm_aligned)
    overlay_rgb = np.stack([norm_hr, norm_aligned, norm_hr], axis=-1)  # Green=Aligned LR, Magenta=HR

    plt.imshow(overlay_rgb)
    plt.title("Alignment Result (Green: Aligned LR / Magenta: HR)")
    plt.show()