import os
import json
from tqdm import tqdm

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from utils.utils import data_transforms
from utils.prompts import init_point_sampling, get_boxes_from_mask


class TestingDataset(Dataset):
    
    def __init__(self, data_path, mode='test', strategy='base', point_num=1, image_size=256):
        """
        Initializes a TestingDataset object.
        Args:
            data_path (str): The path to the data.
            mode (str, optional): The mode of the dataset. Defaults to 'test'.
            point_num (int, optional): The number of points to retrieve. Defaults to 1.
            image_size (int, optional): The size of the image. Defaults to 256.
        """
        self.strategy = strategy
        self.point_num = point_num
        self.image_size = image_size

        json_file = open(os.path.join(data_path, mode, f'label2image_test.json'), "r")
        dataset = json.load(json_file)    
        self.image_paths = list(dataset.values())
        self.label_paths = list(dataset.keys())
      
        self.pixel_mean = [123.675, 116.28, 103.53]
        self.pixel_std = [58.395, 57.12, 57.375]
    
    def __len__(self):
        return len(self.label_paths)
    
    def __getitem__(self, index):
        """
        Retrieves and preprocesses an item from the dataset.
        Args:
            index (int): The index of the item to retrieve.
        Returns:
            dict: A dictionary containing the preprocessed image and associated information.
        """
        image_input = {}
        try:
            image = cv2.imread(self.image_paths[index])
            image = (image - self.pixel_mean) / self.pixel_std
        except:
            print(self.image_paths[index])

        mask_path = self.label_paths[index]
        ori_np_mask = cv2.imread(mask_path, 0)
        
        if ori_np_mask.max() == 255:
            ori_np_mask = ori_np_mask / 255

        assert np.array_equal(ori_np_mask, ori_np_mask.astype(bool)), f"Mask should only contain binary values 0 and 1. {self.label_paths[index]}"

        h, w = ori_np_mask.shape
        ori_mask = torch.tensor(ori_np_mask).unsqueeze(0)

        transforms = data_transforms(self.image_size, h, w)
        augments = transforms(image=image, mask=ori_np_mask)
        image, mask = augments['image'], augments['mask'].to(torch.int64)
        
        # initial prompts
        point_coords, point_labels = init_point_sampling(mask, self.point_num, self.strategy)
        boxes = get_boxes_from_mask(mask, image_size=self.image_size)

        image_input["im_name"] = self.label_paths[index].split('/')[-1]
        image_input["image"] = image
        image_input["label"] = mask.unsqueeze(0)
        image_input["ori_label"] = ori_mask
        image_input["original_size"] = (h, w)
        
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels
        image_input["boxes"] = boxes

        return image_input


if __name__ == "__main__":
    dataset = TestingDataset("/home/yanzhonghao/data/ven/bhx_sammed", mode='test', image_size=1024,)
    print("Dataset:", len(dataset))
    dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=16, shuffle=False, )
    tbar = tqdm((dataloader), total = len(dataloader), leave=False)
    
    for image_input in tbar:   
        print(image_input["im_name"])
        print(image_input["image"].shape, image_input["label"].shape)
        break