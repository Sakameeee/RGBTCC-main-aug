import numpy as np
import os
from glob import glob
import cv2
import json


def generate_data(label_path):
    # 借由 json 文件名读取到 rgb 和 t 图
    rgb_path = label_path.replace('GT', 'RGB').replace('json', 'jpg')
    t_path = label_path.replace('GT', 'T').replace('json', 'jpg')
    # 反转通道，在 OpenCV 中，颜色通道的默认顺序是 BGR
    rgb = cv2.imread(rgb_path)[..., ::-1].copy()
    t = cv2.imread(t_path)[..., ::-1].copy()
    # 图像宽高
    im_h, im_w, _ = rgb.shape
    print('rgb and t shape', rgb.shape, t.shape)
    # 读取 json 文件
    with open(label_path, 'r') as f:
        label_file = json.load(f)
    # 获取 json 中 points 里的数组
    points = np.asarray(label_file['points'])
    # print('points', points.shape)
    # 创建布尔掩码，idx_mask 是一个与 points 等长的布尔数组
    idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h)
    points = points[idx_mask]
    return rgb, t, points


if __name__ == '__main__':

    root_path = 'E:/code/DataSets/RGBT/RGBT-CC-CVPR2021'  # dataset root path
    save_dir = 'E:/code/DataSets/RGBT/RGBT-CC-A-use'

    for phase in ['train', 'val', 'test']:
        sub_dir = os.path.join(root_path, phase)
        sub_save_dir = os.path.join(save_dir, phase)
        if not os.path.exists(sub_save_dir):
            os.makedirs(sub_save_dir)
        # 通配符匹配所有 json 文件
        gt_list = glob(os.path.join(sub_dir, '*json'))
        for gt_path in gt_list:
            name = os.path.basename(gt_path)
            # print('name', name)
            # 获取 翻转之后的 rgb，t 图，和经过蒙版的点集
            rgb, t, points = generate_data(gt_path)
            im_save_path = os.path.join(sub_save_dir, name)
            rgb_save_path = im_save_path.replace('GT', 'RGB').replace('json', 'jpg')
            t_save_path = im_save_path.replace('GT', 'T').replace('json', 'jpg')
            # 写入保存文件
            cv2.imwrite(rgb_save_path, rgb)
            cv2.imwrite(t_save_path, t)
            gd_save_path = im_save_path.replace('json', 'npy')
            np.save(gd_save_path, points)
