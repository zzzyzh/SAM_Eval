import os
import glob
import shutil
import json
from tqdm import tqdm

import cv2
import pydicom
import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image


VAL_VOLUME = [
    "0035",
    "0036",
    "0037",
]

TEST_VOLUME = [
    "0038",
    "0039",
    "0040",
] # reference: https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR

part_atlas = {
    1: 'spleen', 
    2: 'rkid', 
    3: 'lkid', 
    4: "gall", 
    6: 'liver', 
    7: "sto", 
    8: "aorta", 
    11: "pancreas"
}

hashmap = {1:1, 2:2, 3:3, 4:4, 5:0, 6:5, 7:6, 8:7, 9:0, 10:0, 11:8, 12:0, 13:0}
a_min, a_max = -125, 275


def pre_process(target_path, source_path):
    source_images_path = os.path.join(source_path, 'img')
    source_masks_path = os.path.join(source_path, 'label')
    
    for task in ['train', 'val', 'test']:    
        target_images_path = os.path.join(target_path, task, 'images')
        target_npz_path = os.path.join(target_path, task, 'npz')
        target_masks_path = os.path.join(target_path, task, 'masks')
        target_labels_path = os.path.join(target_path, task, 'labels')
        os.makedirs(target_images_path, exist_ok=True)
        os.makedirs(target_npz_path, exist_ok=True)
        os.makedirs(target_masks_path, exist_ok=True)
        os.makedirs(target_labels_path, exist_ok=True)

        mask2image = {}
        label2image = {}
        image2label = {}
        
        for im_name in tqdm(sorted(os.listdir(source_images_path))):
            
            index = im_name[3:7]
            lbl_name = f'label{index}.nii.gz'
            train = False
            if task == 'train' and (index not in VAL_VOLUME and index not in TEST_VOLUME):
                train = True
            if task == 'val' and index in VAL_VOLUME:
                train = True
            if task == 'test' and index in TEST_VOLUME:
                train = True
        
            if not train:
                continue
            
            source_nii = nib.load(os.path.join(source_images_path, im_name)).get_fdata()
            source_mask = nib.load(os.path.join(source_masks_path, lbl_name)).get_fdata()

            source_nii = source_nii.astype(np.float32)
            source_mask = source_mask.astype(np.float32)

            source_nii = np.clip(source_nii, a_min, a_max)

            source_nii = np.transpose(source_nii, (2, 1, 0))  # [D, W, H]
            source_mask = np.transpose(source_mask, (2, 1, 0))
            source_image = (source_nii - source_nii.min()) / (source_nii.max() - source_nii.min()) * 255.0

            for i, (slice_mask, slice_nii, slice_image) in enumerate(zip(source_mask, source_nii, source_image)):
                unique_pixel = np.unique(slice_mask)[1:]

                save = False
                for k in sorted(part_atlas.keys()):
                    if k not in unique_pixel:
                        continue
                    
                    target_name = f'sabs_{index}_{str(i).zfill(3)}'
                    mask = np.zeros_like(slice_mask)
                    mask[slice_mask == k] = 255
                    if np.sum(mask == 255) < 100:
                        continue
                
                    save = True
                    label2image[os.path.join(target_labels_path, f'{target_name}_{part_atlas[k]}.png')] = os.path.join(target_images_path, f'{target_name}.png')
                    cv2.imwrite(os.path.join(target_labels_path, f'{target_name}_{part_atlas[k]}.png'), mask)
            
                if save:        
                    # 保存为 npz
                    np.savez(os.path.join(target_npz_path, f'{target_name}.npz'), image=slice_nii)
                    # 保存为 png
                    cv2.imwrite(os.path.join(target_images_path, f'{target_name}.png'), np.uint8(slice_image)) 
                    # 保存为 mask
                    for raw, new in hashmap.items():
                        slice_mask[slice_mask == raw] = new
                    cv2.imwrite(os.path.join(target_masks_path, f'{target_name}.png'), np.uint8(slice_mask)) 
                    mask2image[os.path.join(target_masks_path, f'{target_name}.png')] = os.path.join(target_images_path, f'{target_name}.png')
        
        for label, image in label2image.items():
            if image not in image2label.keys():
                image2label[image] = [label]
            else:
                image2label[image].append(label)
                image2label[image] = sorted(image2label[image])
        
        with open(os.path.join(target_path, 'label2image_test.json'), 'w', newline='\n') as f:
            json.dump(label2image, f, indent=2)  # 换行显示
        with open(os.path.join(target_path, 'image2label_train.json'), 'w', newline='\n') as f:
            json.dump(image2label, f, indent=2)  # 换行显示
        with open(os.path.join(target_path, 'mask2image.json'), 'w', newline='\n') as f:
            json.dump(mask2image, f, indent=2)  # 换行显示


if __name__ == '__main__':
    source_path = '/home/yanzhonghao/data/abdomen/SABS/RawData/Training'
    target_path = '/home/yanzhonghao/data/abdomen/sabs_sammed'
    
    pre_process(target_path, source_path)
