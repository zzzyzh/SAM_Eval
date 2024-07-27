import os
import glob
import shutil
import json
from tqdm import tqdm

import cv2
import nibabel as nib
import numpy as np
import niftiio as nio
import SimpleITK as sitk


VAL_VOLUME = [
    "0002", "0008", "0022", "0026", "0028", "0012"
]

TEST_VOLUME = [
    "0019", "0003", "0001", "0004", "0015", "0025"
]

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
    imgs = glob.glob(source_path + "/image_*.nii.gz")
    labels = glob.glob(source_path + "/label_*.nii.gz")
    imgs = [ fid for fid in sorted(imgs) ]
    labels = [ fid for fid in sorted(labels) ]
    pids = [pid.split("_")[-1].split(".")[0] for pid in imgs]

    for task in ['train', 'val', 'test']:    
        target_images_path = os.path.join(target_path, task, 'images')
        target_masks_path = os.path.join(target_path, task, 'masks')
        target_labels_path = os.path.join(target_path, task, 'labels')
        os.makedirs(target_images_path, exist_ok=True)
        os.makedirs(target_masks_path, exist_ok=True)
        os.makedirs(target_labels_path, exist_ok=True)

        mask2image = {}
        label2image = {}
        image2label = {}
        
        for img_fid, seg_fid, pid in tqdm(zip(imgs, labels, pids)):

            img_obj = sitk.ReadImage( img_fid )
            seg_obj = sitk.ReadImage( seg_fid )
        
            index = str(pid).zfill(4)
        
            train = False
            if task == 'train' and (index not in VAL_VOLUME and index not in TEST_VOLUME):
                train = True
            if task == 'val' and index in VAL_VOLUME:
                train = True
            if task == 'test' and index in TEST_VOLUME:
                train = True
            if not train:
                continue
            
            source_img = sitk.GetArrayFromImage(img_obj)
            source_mask = sitk.GetArrayFromImage(seg_obj)

            for i, (slice_image, slice_mask) in enumerate(zip(source_img, source_mask)):
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
                    # 保存为 png
                    cv2.imwrite(os.path.join(target_images_path, f'{target_name}.png'), np.uint8(slice_image)) 
                    # 保存为 mask
                    save_mask = np.zeros_like(slice_mask)
                    for k in sorted(part_atlas.keys()):
                        save_mask[slice_mask == k] = hashmap[k]
                    cv2.imwrite(os.path.join(target_masks_path, f'{target_name}.png'), np.uint8(save_mask)) 
                    mask2image[os.path.join(target_masks_path, f'{target_name}.png')] = os.path.join(target_images_path, f'{target_name}.png')
        
        for label, image in label2image.items():
            if image not in image2label.keys():
                image2label[image] = [label]
            else:
                image2label[image].append(label)
                image2label[image] = sorted(image2label[image])
        
        with open(os.path.join(target_path, task, 'label2image_test.json'), 'w', newline='\n') as f:
            json.dump(label2image, f, indent=2)  # 换行显示
        with open(os.path.join(target_path, task, 'image2label_train.json'), 'w', newline='\n') as f:
            json.dump(image2label, f, indent=2)  # 换行显示
        with open(os.path.join(target_path, task, 'mask2image.json'), 'w', newline='\n') as f:
            json.dump(mask2image, f, indent=2)  # 换行显示


def resize_shape(source_path):
    
    for task in ['train', 'val', 'test']:   
        root_path = os.path.join(source_path, task)
        
        for sub in ['images', 'labels', 'masks']:
            sub_path = os.path.join(root_path, sub)

            for im_name in tqdm(sorted(os.listdir(sub_path))):
                image = cv2.imread(os.path.join(sub_path, im_name))
                
                if sub == 'image':
                    resize_image = cv2.resize(image, (512,512), interpolation=cv2.INTER_CUBIC)
                else:
                    resize_image = cv2.resize(image, (512,512), interpolation=cv2.INTER_NEAREST)

                cv2.imwrite(os.path.join(sub_path, im_name), resize_image)
                

if __name__ == '__main__':
    source_path = '../../data/abdomen/SABS/sabs_CT_normalized'
    target_path = '../../data/abdomen/sabs_sammed'
    
    pre_process(target_path, source_path)
    resize_shape(target_path)