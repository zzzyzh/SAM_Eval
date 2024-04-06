import os
import glob
import shutil
import json
from tqdm import tqdm

import cv2
import pydicom
import numpy as np
import pandas as pd
from PIL import Image

    
def pre_data(target_path, source_path):
    target_dicoms_path = os.path.join(target_path, 'dicoms')
    target_images_path = os.path.join(target_path, 'images')
    target_masks_path = os.path.join(target_path, 'masks')
    target_labels_path = os.path.join(target_path, 'labels')

    os.makedirs(target_dicoms_path, exist_ok=True)
    os.makedirs(target_images_path, exist_ok=True)
    os.makedirs(target_masks_path, exist_ok=True)
    os.makedirs(target_labels_path, exist_ok=True)

    source_images_path = os.path.join(source_path, 'images')
    source_masks_path = os.path.join(source_path, 'labels')

    ven = ['right', 'left', 'third', 'fourth']
    mask2image = {}
    label2image = {}
    
    for _, index in enumerate(tqdm(sorted(os.listdir(source_masks_path)))):
        if int(index) > 800:
            break
        
        for im_name in sorted(os.listdir(os.path.join(source_masks_path, index))):
            im_name = im_name.split('.')[0]
            
            source_mask = cv2.imread(os.path.join(source_masks_path, index, f'{im_name}.png'))
            source_mask = cv2.cvtColor(source_mask, cv2.COLOR_BGR2GRAY)
            if np.max(source_mask) == 0:
                continue  # 不包含脑室
            
            source_dcm = pydicom.dcmread(os.path.join(source_images_path, index, f'{im_name}.dcm'))
            # 将像素值转换为 Hounsfield units (HU)
            data = source_dcm.pixel_array
            slope = source_dcm.RescaleSlope
            intercept = source_dcm.RescaleIntercept
            hu_data = slope * data + intercept
            
            # 设置窗宽和窗位
            window_width = 80  # 窗宽
            window_level = 50  # 窗位
            
            CT_min = window_level - window_width / 2
            CT_max = window_level + window_width / 2
            data = np.clip(hu_data, CT_min, CT_max)
            
            # 保存为 dicom
            source_dcm.PixelData = data.astype(np.int16).tobytes()
            source_dcm.save_as(os.path.join(target_dicoms_path, f'{index}_{im_name}.dcm'))
            
            # 保存为 png
            data_min = data.min()
            data_max = data.max()
            source_image = np.uint8(((data - data_min) / (data_max - data_min + 1e-5)) * 255)

            unique_pixel = np.unique(source_mask)[1:]
            if 5 in unique_pixel:
                unique_pixel = unique_pixel[:-1]
            
            save = 0
            for uni in unique_pixel:
                mask = np.zeros_like(source_mask)
                mask[source_mask == uni] = 255
                
                # 寻找轮廓
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                # 循环遍历每个轮廓
                for i, contour in enumerate(contours):
                    target = np.zeros_like(source_mask)
                    cv2.drawContours(target, [contour], -1, 255, thickness=cv2.FILLED)
                    if np.sum(target == 255) < 100:
                        continue
                    save += 1
                    target_name = f'bhx_{index}_{im_name}_{ven[uni-1]}_{i}.png'
                    label2image[os.path.join(target_labels_path, target_name)] = os.path.join(target_images_path,
                                                                                        f'bhx_{index}_{im_name}.png')
                    cv2.imwrite(os.path.join(target_labels_path, target_name), target)
            
            if save > 0:
                target_name = f'bhx_{index}_{im_name}.png'
                cv2.imwrite(os.path.join(target_images_path, target_name), source_image) 
                shutil.copy(os.path.join(source_masks_path, index, f'{im_name}.png'), os.path.join(target_masks_path, target_name))
                mask2image[os.path.join(target_masks_path, target_name)] = os.path.join(target_images_path, target_name)

    with open(os.path.join(target_path, 'label2image_test.json'), 'w', newline='\n') as f:
        json.dump(label2image, f, indent=2)  # 换行显示
    with open(os.path.join(target_path, 'mask2image.json'), 'w', newline='\n') as f:
        json.dump(mask2image, f, indent=2)  # 换行显示

    print('Success')
            

if __name__ == '__main__':
    
    target_path = '/home/yanzhonghao/data/ven/bhx'
    source_path = '/home/yanzhonghao/data/ven/bhx_sammed'
    pre_data(target_path, source_path) 
