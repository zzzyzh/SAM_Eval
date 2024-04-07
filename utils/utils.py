import os
import logging

import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
import albumentations as A
import matplotlib.pyplot as plt


# data
def data_transforms(img_size, ori_h, ori_w):
    transforms = []
    if ori_h < img_size and ori_w < img_size:
        transforms.append(A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)))
    else:
        transforms.append(A.Resize(int(img_size), int(img_size), interpolation=cv2.INTER_NEAREST))
    transforms.append(ToTensorV2(p=1.0))
    return A.Compose(transforms, p=1.)


# save
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    fh = logging.FileHandler(filename)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def draw_boxes(img, boxes):
    img_copy = np.copy(img)
    for box in boxes:
        cv2.rectangle(img_copy, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    return img_copy


def save_masks(save_path, mask_name, preds, gts, image_size, original_size, pad=None, boxes=None, points=None, visual_prompt=True):
    ori_h, ori_w = original_size

    preds = torch.sigmoid(preds)
    preds[preds > 0.5] = int(1)
    preds[preds <= 0.5] = int(0)

    gt = gts.squeeze().cpu().numpy()
    gt = (gt / np.max(gt)) * 255
    mask = preds.squeeze().cpu().numpy()
    mask = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)
    
    if visual_prompt: # visualize the prompt
        if boxes is not None:
            boxes = boxes.squeeze().cpu().numpy()

            x0, y0, x1, y1 = boxes
            if pad is not None:
                x0_ori = int((x0 - pad[1]) + 0.5)
                y0_ori = int((y0 - pad[0]) + 0.5)
                x1_ori = int((x1 - pad[1]) + 0.5)
                y1_ori = int((y1 - pad[0]) + 0.5)
            else:
                x0_ori = int(x0 * ori_w / image_size) 
                y0_ori = int(y0 * ori_h / image_size) 
                x1_ori = int(x1 * ori_w / image_size) 
                y1_ori = int(y1 * ori_h / image_size)

            boxes = [(x0_ori, y0_ori, x1_ori, y1_ori)]
            mask = draw_boxes(mask, boxes)

        if points is not None:
            point_coords, point_labels = points[0].squeeze(0).cpu().numpy(),  points[1].squeeze(0).cpu().numpy()
            point_coords = point_coords.tolist()
            if pad is not None:
                ori_points = [[int((x * ori_w / image_size)) , int((y * ori_h / image_size))]if l==0 else [x - pad[1], y - pad[0]]  for (x, y), l in zip(point_coords, point_labels)]
            else:
                ori_points = [[int((x * ori_w / image_size)) , int((y * ori_h / image_size))] for x, y in point_coords]

            for point, label in zip(ori_points, point_labels):
                x, y = map(int, point)
                color = (0, 255, 0) if label == 1 else (0, 0, 255)
                mask[y, x] = color
                cv2.drawMarker(mask, (x, y), color, markerType=cv2.MARKER_CROSS , markerSize=7, thickness=2) 
    
    plt.figure(figsize=(6, 3)) # 设置画布大小
    plt.subplot(1, 2, 1)  # 1行2列的子图中的第1个
    plt.imshow(np.uint8(gt), cmap='gray')  # 使用灰度颜色映射
    plt.title('Ground Truth')  # 设置标题
    plt.axis('off')  # 关闭坐标轴
    
    plt.subplot(1, 2, 2)  # 1行2列的子图中的第2个
    plt.imshow(np.uint8(mask))  # 使用灰度颜色映射
    plt.title('Prediction')  # 设置标题
    plt.axis('off')  # 关闭坐标轴
    
    plt.savefig(os.path.join(save_path, mask_name))
    plt.close()
    