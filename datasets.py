import json

import cv2
import torch
from torch.utils.data import Dataset

from utils.utils import data_transforms
from utils.prompts import init_point_sampling, get_boxes_from_mask


class SAMEvalDataset(Dataset):
    def __init__(self, dataset_name, image_size=1024, strategy='base', point_num=1, max_pixel=0):
        """
        Initializes a SAMEvalDataset object.
        Args:
            image_size (int, optional): The size of the image. Defaults to 1024.
        """
        
        self.image_size = image_size
        self.strategy = strategy
        self.point_num = point_num
        self.max_pixel = max_pixel
        
        json_file = open(f'dataset_split/{dataset_name}.json', "r")
        dataset = json.load(json_file)    

        self.image_paths = list(dataset.values())
        self.label_paths = list(dataset.keys())

        self.pixel_mean = [123.675, 116.28, 103.53]
        self.pixel_std = [58.395, 57.12, 57.375]
        # self.pixel_mean = [0., 0., 0.]
        # self.pixel_std = [1., 1., 1.]
    
    def __len__(self):
        # return len(self.label_paths)
        return int(len(self.label_paths) * 0.05)  # For Test
    
    def __getitem__(self, index):
        """
        Retrieves and preprocesses an item from the dataset.
        Args:
            index (int): The index of the item to retrieve.
        Returns:
            dict: A dictionary containing the preprocessed image and associated information.
        """
        image = cv2.imread(self.image_paths[index])
        image = (image - self.pixel_mean) / self.pixel_std
        ori_label = cv2.imread(self.label_paths[index], 0)
        if ori_label.max() == 255:
            ori_label = ori_label / 255
        h, w = ori_label.shape

        transforms = data_transforms(self.image_size, h, w)
        augments = transforms(image=image, mask=ori_label)
        image, label = augments['image'], augments['mask'].to(torch.int64)
        ori_label = torch.tensor(ori_label).unsqueeze(0)

        # initial prompts
        point_coords, point_labels = init_point_sampling(label, self.point_num, self.strategy)
        boxes = get_boxes_from_mask(label, strategy=self.strategy, image_size=self.image_size, max_pixel=self.max_pixel)

        sample = {}
        sample["im_name"] = self.label_paths[index].split('/')[-1]
        sample["image"] = image
        sample["label"] = label.unsqueeze(0)
        sample["ori_label"] = ori_label
        sample["original_size"] = (h, w)
        
        sample["point_coords"] = point_coords
        sample["point_labels"] = point_labels
        sample["boxes"] = boxes

        return sample


if __name__ == "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader

    dataset = SAMEvalDataset(
        dataset_name='AbdomenAtlas1.1',
        image_size=256,
        strategy='base',
        point_num=1,
        max_pixel=0
    )
    dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=16, shuffle=False, )
    tbar = tqdm((dataloader), total = len(dataloader), leave=False)
    for image_input in tbar:   
        print(image_input["im_name"][0][5:-6])
        print(image_input["image"].shape, image_input["label"].shape)
        break
    
    