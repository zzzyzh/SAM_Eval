import os
import glob
import json
from tqdm import tqdm

import cv2
import nibabel as nib
import numpy as np

from dataset_map import CLASS_MAP


def resize_image(image, target_size, is_mask=False):
    """
    Resize 3D volume using scipy.ndimage.zoom with customizable interpolation order.

    Parameters:
        image (np.ndarray): Input 3D volume (H, W, D).
        target_size (tuple): (target_H, target_W, target_D).

    Returns:
        np.ndarray: Resized volume.
    """
    target_H, target_W = target_size
    H, W = image.shape

    pad_value = 0
    interp_method = cv2.INTER_NEAREST if is_mask else cv2.INTER_CUBIC

    if H <= target_H and W <= target_W:
        # Case 1: padding if smaller
        pad_H = target_H - H
        pad_W = target_W - W
        pad_top = pad_H // 2
        pad_bottom = pad_H - pad_top
        pad_left = pad_W // 2
        pad_right = pad_W - pad_left

        resize_image = np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=pad_value
        )
    else:
        # Case 2: resize if larger
        resize_image = cv2.resize(image, (target_W, target_H), interpolation=interp_method)

    return resize_image


def prepare_abdomen_atlas_1_1(raw_path, target_path):
    HU_min = -200
    HU_max = 250
    class_map_0 = CLASS_MAP['AbdomenAtlas1.0']
    class_map_1 = CLASS_MAP['AbdomenAtlas1.1']
    
    data_split_0 = {}
    data_split_1 = {}
    data_list = sorted(glob.glob(os.path.join(raw_path, "BDMAP_*")))
    data_list = data_list[ : 50] # For Test
    for data_path in tqdm(data_list):
        data_name = os.path.basename(data_path)
        image_path = os.path.join(data_path, "ct.nii.gz")
        mask_path = os.path.join(data_path, "combined_labels.nii.gz")
        
        image_nii = nib.load(image_path)
        mask_nii = nib.load(mask_path)
        
        image_data = image_nii.get_fdata()
        mask_data = mask_nii.get_fdata()

        image_data = np.clip(image_data, HU_min, HU_max)
        image_data = (image_data - HU_min) / (HU_max - HU_min) * 255
        image_data = np.uint8(image_data)
                
        target_image_path = os.path.join(target_path, data_name, 'images')
        target_mask_path = os.path.join(target_path, data_name, 'masks')
        os.makedirs(target_image_path, exist_ok=True)
        os.makedirs(target_mask_path, exist_ok=True)
        
        for slice_indx in tqdm(range(image_data.shape[2])):
            image = image_data[:, :, slice_indx]
            mask = mask_data[:, :, slice_indx]
            
            image = np.rot90(image, k=1, axes=(0, 1))
            mask = np.rot90(mask, k=1, axes=(0, 1))

            image = resize_image(image, (512, 512))
            mask = resize_image(mask, (512, 512), is_mask=True)
            
            target_image_slice_path = os.path.join(target_image_path, f"{slice_indx:04d}.png")
            cv2.imwrite(target_image_slice_path, image)

            mask_list = np.unique(mask)
            mask_list = sorted(mask_list)[1:]
            for mask_index in mask_list:
                mask_name = class_map_1[int(mask_index) - 1]
                mask_class = np.zeros_like(mask)
                mask_class[mask == mask_index] = 255
                
                mask_part_list = []
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_class.astype(np.uint8), connectivity=8)
                for i in range(1, num_labels):
                    # Follow SAM-Med2D: https://github.com/OpenGVLab/SAM-Med2D
                    if stats[i, cv2.CC_STAT_AREA] >= 100:
                        single_mask = np.where(labels == i, 255, 0).astype(np.uint8)
                        mask_part_list.append(single_mask)
                    # single_mask = np.where(labels == i, 255, 0).astype(np.uint8)
                    # mask_part_list.append(single_mask)

                for i in range(len(mask_part_list)):
                    mask_part = mask_part_list[i]
                    mask_part = np.uint8(mask_part)
                    target_mask_slice_path = os.path.join(target_mask_path, f"{slice_indx:04d}_{mask_name}_{i}.png")
                    cv2.imwrite(target_mask_slice_path, mask_part)
                    
                    data_split_1[target_mask_slice_path] = target_image_slice_path
                    if int(mask_index) - 1 <= 8:
                        data_split_0[target_mask_slice_path] = target_image_slice_path
    
    with open('dataset_split/AbdomenAtlas1.0.json', 'w') as f:
        json.dump(data_split_0, f, indent=4)

    with open('dataset_split/AbdomenAtlas1.1.json', 'w') as f:
        json.dump(data_split_1, f, indent=4)


if __name__ == "__main__":
    raw_path = '/home/yanzhonghao/data/dataset_hub/AbdomenAtlas1.1'
    target_path = '/home/yanzhonghao/data/datasets/AbdomenAtlas1.1'
    os.makedirs(target_path, exist_ok=True)
    
    prepare_abdomen_atlas_1_1(raw_path, target_path)
    