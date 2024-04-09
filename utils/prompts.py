import os
import logging

import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
import albumentations as A
import torch
import torch.nn as nn
from torch.nn import functional as F
from skimage.measure import label, regionprops
from matplotlib import pyplot as plt


# Generating prompts methods
def setting_prompt_none(batched_input):
    batched_input["point_coords"] = None
    batched_input["point_labels"] = None
    batched_input["boxes"] = None
    return batched_input


# Boxes
def get_boxes_from_mask(mask, strategy='base', image_size=256, box_num=1, std=0.1, max_pixel=5):
    """
    Args:
        mask: Mask, can be a torch.Tensor or a numpy array of binary mask.
        box_num: Number of bounding boxes, default is 1.
        std: Standard deviation of the noise, default is 0.1.
        max_pixel: Maximum noise pixel value, default is 5.
    Returns:
        noise_boxes: Bounding boxes after noise perturbation, returned as a torch.Tensor.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
        
    label_img = label(mask)
    regions = regionprops(label_img)

    # Iterate through all regions and get the bounding box coordinates
    boxes = [tuple(region.bbox) for region in regions]
    centroids = [tuple(region.centroid) for region in regions]

    # If the generated number of boxes is greater than the number of categories,
    # sort them by region area and select the top n regions
    if len(boxes) >= box_num:
        sorted_regions = sorted(regions, key=lambda x: x.area, reverse=True)[:box_num]
        boxes = [tuple(region.bbox) for region in sorted_regions]
        centroids = [tuple(region.centroid) for region in sorted_regions]
        
    # If the generated number of boxes is less than the number of categories,
    # duplicate the existing boxes
    elif len(boxes) < box_num:
        num_duplicates = box_num - len(boxes)
        boxes += [boxes[i % len(boxes)] for i in range(num_duplicates)]
        centroids += [centroids[i % len(centroids)] for i in range(num_duplicates)]

    # Perturb each bounding box with noise
    noise_boxes = []
    for box, centroid in zip(boxes, centroids):
        y0, x0, y1, x1  = box
        if strategy == 'base':
            width, height = abs(x1 - x0), abs(y1 - y0)
        elif 'square' in strategy:
            center_x, center_y = centroid[1], centroid[0]
            if 'max' in strategy:
                width, height = max(abs(x1-x0), abs(y1-y0)), max(abs(x1-x0), abs(y1-y0))
            elif 'min' in strategy:
                width, height = min(abs(x1-x0), abs(y1-y0)), min(abs(x1-x0), abs(y1-y0))
            x0 = int(center_x - width/2)
            x1 = int(center_x + width/2)
            y0 = int(center_y - height/2)
            y1 = int(center_y + height/2)
        # Calculate the standard deviation and maximum noise value
        noise_std = min(width, height) * std
        max_noise = min(max_pixel, int(noise_std * max_pixel))
        # Add random noise to each coordinate
        try:
            noise_x = np.random.randint(-max_noise, max_noise)
        except:
            noise_x = 0
        try:
            noise_y = np.random.randint(-max_noise, max_noise)
        except:
            noise_y = 0
        x0, y0 = x0 + noise_x, y0 + noise_y
        x1, y1 = x1 + noise_x, y1 + noise_y
        noise_boxes.append((x0, y0, x1, y1))
    return torch.as_tensor(noise_boxes, dtype=torch.float)


# Points
def init_point_sampling(mask, get_point=1, strategy='base'):
    """
    Initialization samples points from the mask and assigns labels to them.
    Args:
        mask (torch.Tensor): Input mask tensor.
        num_points (int): Number of points to sample. Default is 1.
        strategy (str): Strategy to choose points.
    Returns:
        coords (torch.Tensor): Tensor containing the sampled points' coordinates (x, y).
        labels (torch.Tensor): Tensor containing the corresponding labels (0 for background, 1 for foreground).
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
        
    # Get coordinates of black/white pixels
    fg_coords = np.argwhere(mask == 1)[:,::-1]
    bg_coords = np.argwhere(mask == 0)[:,::-1]

    fg_size = len(fg_coords)
    bg_size = len(bg_coords)

    if get_point == 1:
        if fg_size > 0: 
            if strategy == 'far':
                # Compute the distance transform of the binary mask
                dist_transform = cv2.distanceTransform(np.uint8(mask), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
                # Find the location of the point with maximum distance value
                max_dist = np.max(dist_transform)
                fg_coords = np.argwhere(dist_transform == max_dist)[:,::-1]
                fg_size = len(fg_coords)
            index = np.random.randint(fg_size)
            fg_coord = fg_coords[index]
            label = 1
        else:
            index = np.random.randint(bg_size)
            fg_coord = bg_coords[index]
            label = 0
        
        return torch.as_tensor([fg_coord.tolist()], dtype=torch.float), torch.as_tensor([label], dtype=torch.int)
    else:
        num_fg = get_point // 2
        num_bg = get_point - num_fg
        fg_indices = np.random.choice(fg_size, size=num_fg, replace=True)
        bg_indices = np.random.choice(bg_size, size=num_bg, replace=True)
        fg_coords = fg_coords[fg_indices]
        bg_coords = bg_coords[bg_indices]
        coords = np.concatenate([fg_coords, bg_coords], axis=0)
        labels = np.concatenate([np.ones(num_fg), np.zeros(num_bg)]).astype(int)
        indices = np.random.permutation(get_point)
        coords, labels = torch.as_tensor(coords[indices], dtype=torch.float), torch.as_tensor(labels[indices], dtype=torch.int)
        return coords, labels


def generate_point(masks, labels, low_res_masks, batched_input, strategy='base', point_num=1, image_size=1024,):
    low_res_masks_clone = low_res_masks.clone()
    low_res_masks_logist = torch.sigmoid(low_res_masks_clone)
    
    masks_clone = masks.clone()
    masks_sigmoid = torch.sigmoid(masks_clone)
    masks_binary = (masks_sigmoid > 0.5).float()
    
    labels = labels if masks.shape == labels.shape else F.interpolate(labels, size=(image_size, image_size), mode="nearest").float()
    
    if strategy == 'base':
        points, point_labels = select_random_points(masks_binary, labels, point_num=point_num, image_size=image_size)
    elif strategy == 'far':
        """
            Find error mask
            Find the point where the error mask is furthest from the boundary
        """
        points, point_labels = [], []
        for j in range(labels.shape[0]):
            pred, gt = masks_binary[j].data.cpu().numpy().squeeze(0), labels[j].data.cpu().numpy().squeeze(0)
            error_mask = np.uint8(np.bitwise_xor(np.uint8(pred), np.uint8(gt)))
            fg_point = get_max_dist_point(error_mask)
            points.append([(fg_point[0], fg_point[1])])
            if np.sum(gt[fg_point[1]][fg_point[0]]) != 1:
                point_labels.append([0])
            else:
                point_labels.append([1])
        points = np.array(points)
        point_labels = np.array(point_labels)
    elif strategy == 'm_area':
        """
            Find error mask
            Compare the area of the false negative and false positive masks
            Find the point where the area is furthest from the boundary
        """
        points, point_labels = [], []
        for j in range(labels.shape[0]):
            pred, gt = masks_binary[j].data.cpu().numpy().squeeze(0), labels[j].data.cpu().numpy().squeeze(0)
            false_positive = np.uint8(np.logical_and(pred == 1, gt == 0))
            false_negative = np.uint8(np.logical_and(pred == 0, gt == 1))
            if np.sum(false_positive) >= np.sum(false_negative):
                fg_point = get_max_dist_point(false_positive)
                points.append([(fg_point[0], fg_point[1])])
                point_labels.append([0])
            else:
                fg_point = get_max_dist_point(false_negative)
                points.append([(fg_point[0], fg_point[1])])
                point_labels.append([1])
        points = np.array(points)
        point_labels = np.array(point_labels)
            
    batched_input["mask_inputs"] = low_res_masks_logist
    batched_input["point_coords"] = torch.as_tensor(points)
    batched_input["point_labels"] = torch.as_tensor(point_labels)
    batched_input["boxes"] = None
    return batched_input


def select_random_points(pred, gt, point_num=9, image_size=256):
    """
    Selects random points from the predicted and ground truth masks and assigns labels to them.
    Args:
        pred (torch.Tensor): Predicted mask tensor.
        gt (torch.Tensor): Ground truth mask tensor.
        point_num (int): Number of random points to select. Default is 9.
    Returns:
        batch_points (np.array): Array of selected points coordinates (x, y) for each batch.
        batch_labels (np.array): Array of corresponding labels (0 for background, 1 for foreground) for each batch.
    """
    pred, gt = pred.data.cpu().numpy(), gt.data.cpu().numpy()
    error = np.zeros_like(pred)
    error[pred != gt] = 1

    # error = np.logical_xor(pred, gt)
    batch_points = []
    batch_labels = []
    for j in range(error.shape[0]):
        one_pred = pred[j].squeeze(0)
        one_gt = gt[j].squeeze(0)
        one_erroer = error[j].squeeze(0)

        indices = np.argwhere(one_erroer == 1)
        if indices.shape[0] > 0:
            selected_indices = indices[np.random.choice(indices.shape[0], point_num, replace=True)]
        else:
            indices = np.random.randint(0, image_size, size=(point_num, 2))
            selected_indices = indices[np.random.choice(indices.shape[0], point_num, replace=True)]
        selected_indices = selected_indices.reshape(-1, 2)

        points, labels = [], []
        for i in selected_indices:
            x, y = i[0], i[1]
            if one_pred[x,y] == 0 and one_gt[x,y] == 1:
                label = 1
            # elif one_pred[x,y] == 1 and one_gt[x,y] == 0:
            #     label = 0
            else:
                label = 0
            points.append((y, x))   # Negate the coordinates
            labels.append(label)

        batch_points.append(points)
        batch_labels.append(labels)
    return np.array(batch_points), np.array(batch_labels)


def get_max_dist_point(mask):
    # Compute the distance transform of the binary mask
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    # Find the location of the point with maximum distance value
    max_dist = np.max(dist_transform)
    cY, cX = np.where(dist_transform == max_dist)
    random_idx = np.random.randint(0, len(cX))
    point = (int(cX[random_idx]), int(cY[random_idx])) # (x, y) coordinates

    return point

