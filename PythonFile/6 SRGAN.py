import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import cv2
import numpy as np
import random


# --- 1. 생성자 (Generator): 이미지를 만드는 화가 ---
# 기존 SRCNN보다 훨씬 깊고 강력한 ResNet 구조를 사용합니다.
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv2(self.prelu(self.bn1(self.conv1(x))))
        return x + self.bn2(residual)  # 입력값을 다시 더해줌 (ResNet 핵심)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # 입력층
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        # 허리 (Residual Blocks) - 5개 정도 쌓음
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(5)])

        # 중간 정리
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        # 출력층 (자글자글한 디테일 생성)
        self.conv_out = nn.Conv2d(64, 1, kernel_size=9, padding=4)

    def forward(self, x):
        block1 = self.block1(x)
        res = self.res_blocks(block1)
        block2 = self.block2(res)
        return torch.tanh(self.conv_out(block1 + block2))  # -1 ~ 1 사이 값 출력


# --- 2. 판별자 (Discriminator): 가짜를 찾아내는 경찰 ---
# 이미지가 "자글자글한지 매끈한지"를 판단합니다.
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
            if bn: block.append(nn.BatchNorm2d(out_filters))
            return block

        self.model = nn.Sequential(
            *discriminator_block(1, 64, bn=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, 1),
            nn.Sigmoid()  # 0(가짜) ~ 1(진짜) 확률 출력
        )

    def forward(self, img):
        return self.model(img).view(-1, 1)


class SimsDataset(Dataset):
    def __init__(self, lr_path, hr_path, matrix_path, mass_pairs, patch_size=64, iterations_per_epoch=1000):
        self.lr_images = []
        self.hr_images = []
        self.patch_size = patch_size
        self.iterations = iterations_per_epoch

        H = np.load(matrix_path)

        print("데이터 로드 중 (GAN 학습용 -1 ~ 1 정규화)...")
        with h5py.File(lr_path, 'r') as f_lr, h5py.File(hr_path, 'r') as f_hr:
            for idx, (lr_name, hr_name) in enumerate(mass_pairs):
                if lr_name not in f_lr or hr_name not in f_hr: continue

                lr_raw = f_lr[lr_name][:]
                hr_raw = f_hr[hr_name][:]
                h, w = hr_raw.shape

                try:
                    # Upscale & Align
                    lr_upscaled = cv2.resize(lr_raw, (w, h), interpolation=cv2.INTER_CUBIC)
                    lr_aligned = cv2.warpPerspective(lr_upscaled, H, (w, h))

                    # --- [중요] GAN용 정규화 (-1 ~ 1 범위) ---
                    def normalize_gan(d):
                        vmax = np.percentile(d, 99.5)  # 핫스팟 제거 강화
                        if vmax == 0: vmax = d.max()
                        d = np.clip(d, 0, vmax)
                        d = d / vmax  # 0 ~ 1
                        return (d * 2.0 - 1.0).astype(np.float32)  # -1 ~ 1 변환

                    # 블러링(Blur) 절대 하지 않음! (노이즈 살리기 위해)
                    self.lr_images.append(normalize_gan(lr_aligned))
                    self.hr_images.append(normalize_gan(hr_raw))

                except:
                    continue
        print("✅ 데이터 준비 완료!")

    def __len__(self):
        return self.iterations

    def __getitem__(self, idx):
        img_idx = random.randint(0, len(self.lr_images) - 1)
        lr_img = self.lr_images[img_idx]
        hr_img = self.hr_images[img_idx]

        h, w = lr_img.shape
        y = random.randint(0, h - self.patch_size)
        x = random.randint(0, w - self.patch_size)

        patch_in = lr_img[y:y + self.patch_size, x:x + self.patch_size]
        patch_tar = hr_img[y:y + self.patch_size, x:x + self.patch_size]

        return torch.from_numpy(patch_in[np.newaxis, ...]), torch.from_numpy(patch_tar[np.newaxis, ...])


def train_gan():
    # --- 설정 ---
    LR_FILE = 'LowMassResolution.hdf5'
    HR_FILE = 'HighMassResolution.hdf5'
    MATRIX_FILE = 'alignment_matrix.npy'
    MASS_PAIRS = [
        # --- Total Ion ---
        ('20251118001 (0) - total u', '20251118005 (0) - total u'),

        # --- Low Mass Range ---
        ('20251118001 (10) - 196.96 u u', '20251118005 (13) - 196.96 u u'),
        # ('20251118001 (11) - 394.93 u u', '20251118005 (14) - 394.93 u u'),
        # ('20251118001 (12) - 27.98 u u', '20251118005 (15) - 27.98 u u'),
        # ('20251118001 (13) - 43.97 u u', '20251118005 (16) - 43.97 u u'),
        # ('20251118001 (14) - 59.97 u u', '20251118005 (17) - 59.97 u u'),
        # ('20251118001 (15) - 16.00 u u', '20251118005 (18) - 16.00 u u'),

        # --- Chemical Formulas ---
        # ('20251118001 (16) - OH- u', '20251118005 (19) - OH- u'),
        # ('20251118001 (17) - C_2H- u', '20251118005 (20) - C_2H- u'),
        ('20251118001 (18) - CN- u', '20251118005 (21) - CN- u'),
        ('20251118001 (19) - Cl- u', '20251118005 (22) - Cl- u'),
        ('20251118001 (20) - ^37Cl- u', '20251118005 (23) - ^37Cl- u'),
        ('20251118001 (21) - CNO- u', '20251118005 (24) - CNO- u'),
        ('20251118001 (24) - C_2H_3O_2- u', '20251118005 (27) - C_2H_3O_2- u'),
        # ('20251118001 (25) - CHSO- u', '20251118005 (28) - CHSO- u'),
        # ('20251118001 (26) - SO_2- u', '20251118005 (29) - SO_2- u'),
        ('20251118001 (28) - SO_3- u', '20251118005 (31) - SO_3- u'),
        ('20251118001 (29) - C_2H_2Cl_2- u', '20251118005 (32) - C_2H_2Cl_2- u'),
        ('20251118001 (30) - C_2H_3Cl_2- u', '20251118005 (33) - C_2H_3Cl_2- u'),
        ('20251118001 (31) - Si_3CH_3- u', '20251118005 (34) - Si_3CH_3- u'),
        ('20251118001 (32) - I- u', '20251118005 (35) - I- u'),
        # ('20251118001 (33) - Si_2CH_5S_2- u', '20251118005 (36) - Si_2CH_5S_2- u'),
        # ('20251118001 (34) - C_3HO_5F_2- u', '20251118005 (37) - C_3HO_5F_2- u'),
        # ('20251118001 (37) - InO_3H- u', '20251118005 (40) - InO_3H- u'),
        # ('20251118001 (38) - InH_2O_3- u', '20251118005 (41) - InH_2O_3- u'),

        # --- High Mass Range ---
        # ('20251118001 (42) - 154.89 u u', '20251118005 (45) - 154.89 u u'),
        # ('20251118001 (43) - C_6H_5PO_4- u', '20251118005 (46) - C_6H_5PO_4- u'),
        ('20251118001 (50) - 248.95 u u', '20251118005 (52) - 248.95 u u'),
        ('20251118001 (51) - 257.94 u u', '20251118005 (53) - 257.94 u u'),
        ('20251118001 (52) - 266.91 u u', '20251118005 (54) - 266.91 u u'),
        ('20251118001 (53) - 268.90 u u', '20251118005 (55) - 268.90 u u'),
        # ('20251118001 (54) - 270.90 u u', '20251118005 (56) - 270.90 u u'),
        # ('20251118001 (55) - 273.93 u u', '20251118005 (57) - 273.93 u u'),
        # ('20251118001 (56) - 328.90 u u', '20251118005 (58) - 328.90 u u'),
        # ('20251118001 (57) - 330.90 u u', '20251118005 (59) - 330.90 u u'),
        # ('20251118001 (58) - 358.84 u u', '20251118005 (60) - 358.84 u u'),
        # ('20251118001 (59) - 393.92 u u', '20251118005 (61) - 393.92 u u'),
        # ('20251118001 (60) - 390.90 u u', '20251118005 (62) - 390.90 u u'),
        # ('20251118001 (61) - 419.93 u u', '20251118005 (63) - 419.93 u u'),
        ('20251118001 (62) - 428.90 u u', '20251118005 (64) - 428.90 u u'),
        # ('20251118001 (63) - 498.85 u u', '20251118005 (65) - 498.85 u u'),
        # ('20251118001 (64) - 500.85 u u', '20251118005 (66) - 500.85 u u'),
        # ('20251118001 (65) - 489.88 u u', '20251118005 (67) - 489.88 u u'),
        # ('20251118001 (66) - 480.89 u u', '20251118005 (68) - 480.89 u u'),
        # ('20251118001 (67) - 520.83 u u', '20251118005 (69) - 520.83 u u'),
        # ('20251118001 (68) - 502.84 u u', '20251118005 (70) - 502.84 u u'),
        # ('20251118001 (69) - 590.89 u u', '20251118005 (71) - 590.89 u u'),
        # '20251118001 (70) - 660.83 u u', '20251118005 (72) - 660.83 u u'),
        # ('20251118001 (71) - 692.81 u u', '20251118005 (73) - 692.81 u u'),
        # ('20251118001 (72) - 752.78 u u', '20251118005 (74) - 752.78 u u'),
    ]

    BATCH_SIZE = 16
    EPOCHS = 100  # GAN은 학습이 오래 걸립니다
    LR_G = 0.0001
    LR_D = 0.0004  # 판별자가 조금 더 빨리 배우게 설정

    if torch.backends.mps.is_available():
        DEVICE = torch.device('mps')
    elif torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')

    # 데이터셋 & 모델 준비
    dataset = SimsDataset(LR_FILE, HR_FILE, MATRIX_FILE, MASS_PAIRS, patch_size=64)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    generator = Generator().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)

    # 최적화 도구 2개 필요
    optimizer_G = optim.Adam(generator.parameters(), lr=LR_G)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR_D)

    # 손실 함수
    criterion_GAN = nn.MSELoss()  # 진짜냐 가짜냐 판단 (Adversarial Loss)
    criterion_content = nn.L1Loss()  # 내용은 비슷해야 함 (Content Loss)

    print("--- GAN 학습 시작 (Texture 복원 모드) ---")

    for epoch in range(EPOCHS):
        for i, (imgs_lr, imgs_hr) in enumerate(dataloader):
            imgs_lr = imgs_lr.to(DEVICE)
            imgs_hr = imgs_hr.to(DEVICE)

            # 진짜/가짜 라벨 생성
            valid = torch.ones(imgs_lr.size(0), 1, device=DEVICE, requires_grad=False)
            fake = torch.zeros(imgs_lr.size(0), 1, device=DEVICE, requires_grad=False)

            # -----------------
            #  1. Generator 학습
            # -----------------
            optimizer_G.zero_grad()

            # 가짜 이미지 생성
            gen_hr = generator(imgs_lr)

            # 판별자 속이기 (내 이미지가 진짜(1)라고 판단하게 만들어라!)
            loss_gan = criterion_GAN(discriminator(gen_hr), valid)

            # 내용은 원본이랑 비슷해야 함 (너무 엉뚱한 그림 그리지 마라)
            loss_content = criterion_content(gen_hr, imgs_hr)

            # 최종 Generator Loss (Texture: 0.01 / Structure: 1.0 비율)
            # 이 비율을 조절해서 노이즈를 얼마나 세게 넣을지 결정합니다.
            loss_G = loss_content + 1e-3 * loss_gan

            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            #  2. Discriminator 학습
            # ---------------------
            optimizer_D.zero_grad()

            # 진짜 이미지는 진짜(1)로 판별
            loss_real = criterion_GAN(discriminator(imgs_hr), valid)
            # 가짜 이미지는 가짜(0)로 판별
            loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)

            loss_D = (loss_real + loss_fake) / 2

            loss_D.backward()
            optimizer_D.step()

        print(f"[Epoch {epoch + 1}/{EPOCHS}] Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), f'srgan_generator_epoch_{epoch + 1}.pth')

    return generator


# 실행
train_gan()