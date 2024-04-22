import os
from glob import glob
import cv2
import numpy as np
import torch
import torchvision

def mixup_data(image1, image2, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    lam = 0.5
    image1 = lam * image1 + (1 - lam) * image2

    return image1


def enhance_image(image, saturation_scale=1.5, contrast_scale=1.5):
    image = image.transpose(0, 2).transpose(1, 2)

    # 对图像进行饱和度和对比度增强
    enhanced_image = torchvision.transforms.Grayscale(num_output_channels=3)(image)

    enhanced_image = enhanced_image.transpose(0, 2).transpose(0, 1)

    return enhanced_image


def blur_image(image, kernel_size=5):
    image = image.transpose(0, 2).transpose(1, 2)

    # 对图像进行模糊处理
    blurred_image = torchvision.transforms.GaussianBlur(kernel_size, sigma=(0.5, 2.0))(image)

    blurred_image = blurred_image.transpose(0, 2).transpose(0, 1)

    return blurred_image


def YOCO(images):
    h = images.shape[0]
    w = images.shape[1]
    images = torch.cat((enhance_image(images[:, 0:int(w/2), :]), blur_image(images[:, int(w/2):w, :])), dim=1) if \
    torch.rand(1) > 0.5 else torch.cat((enhance_image(images[0:int(h/2), :, :]), blur_image(images[int(h/2):h, :, :])), dim=0)
    return images


if __name__ == "__main__":
    # image_path = "E:/code/DataSets/test/1162_A.jpg"
    # aug = cv2.imread(image_path)
    # # 计算每个通道的像素值的均值
    # mean = np.mean(aug, axis=(0, 1))
    # # 计算每个通道的像素值的标准差
    # std = np.std(aug, axis=(0, 1))
    # # 计算最小值和最大值
    # min_val = np.min(aug)
    # max_val = np.max(aug)
    # mean = (mean - min_val) / (max_val - min_val)
    # std = std / (max_val - min_val)
    # print("Mean:", mean)
    # print("Standard Deviation:", std)
    image_path = "E:/code/DataSets/RGBT/RGBT-CC-A-sue"
    for phase in ['new_train_224', 'new_val_224', 'new_test_224']:
        sub_dir = os.path.join(image_path, phase)
        # 通配符匹配所有 json 文件
        gt_list = glob(os.path.join(sub_dir, '*npy'))
        for gt_path in gt_list:
            rgb_path = gt_path.replace('.npy', '.jpg').replace('GT', 'RGB')
            t_path = gt_path.replace('.npy', '.jpg').replace('GT', 'T')
            aug_path = gt_path.replace('.npy', '.jpg').replace('GT', 'A')
            rgb = cv2.imread(rgb_path).copy()
            t = cv2.imread(t_path).copy()
            aug = YOCO(torch.from_numpy(mixup_data(rgb, t))).numpy()
            print(aug_path)
            cv2.imwrite(aug_path, aug)
