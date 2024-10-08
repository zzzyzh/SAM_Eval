import os
import datetime
import random
import json
from datetime import datetime

import argparse
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from segment_anything import sam_model_registry
from sam_med2d import sam_model_registry as sam_med2d_registry
from dataset import TestingDataset
from utils.utils import get_logger, save_masks
from utils.prompts import generate_point, get_max_dist_point, train_lr_model
from utils.metrics import FocalDiceloss_IoULoss, seg_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    # set up model
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--encoder_adapter", type=bool, default=True, help="use adapter")
    parser.add_argument("--sam_mode", type=str, default="sam", choices=['sam', 'sam_med2d', 'med_sam'], help="sam mode")
    parser.add_argument("--model_type", type=str, default="vit_b", choices=['vit_b', 'vit_h'], help="model type of sam")
    parser.add_argument("--sam_checkpoint", type=str, default="../../data/experiments/weights/sam_vit_b_01ec64.pth", help="sam checkpoint")
    parser.add_argument("--multimask", type=bool, default=True, help="ouput multimask")

    # test settings
    parser.add_argument("--run_name", type=str, default="sam_eval", help="repo name")
    parser.add_argument("--root_path", type=str, default="../../data", help="root path")
    parser.add_argument("--save_path", type=str, default="../../data/experiments", help="root path")
    parser.add_argument("--task", type=str, default="ven", choices=['ven', 'abdomen'], help="task name")
    parser.add_argument("--dataset", type=str, default="bhx_sammed", choices=['bhx_sammed', 'sabs_sammed'], help="dataset name")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="num workers")
    parser.add_argument("--image_size", type=int, default=1024, help="image_size")
    parser.add_argument("--max_pixel", type=int, default=0, help="standard pixel")
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice', 'hd95'], help="metrics")
    parser.add_argument("--visual_pred", type=bool, default=False, help="whether to visualize the prediction")
    parser.add_argument("--visual_prompt", type=bool, default=False, help="whether to visualize the prompts")    
    parser.add_argument("--scale", type=float, default=0.1)


    # prompt settings
    parser.add_argument("--prompt", type=str, default='point', choices=['point', 'box', 'fssp'], help = "prompt way")
    parser.add_argument("--mask_prompt", type=bool, default=False, help = "whether to use previous prediction as mask prompts") # If using only one point works well, consider setting it to True
    parser.add_argument("--strategy", type=str, default='base', help = "strategy of each prompt")
    '''
        point: ['base', 'far', 'm_area']
        box: ['base', 'square_max', 'square_min']
    '''
    parser.add_argument("--iter_point", type=int, default=1, help="iter num") 
    parser.add_argument("--point_num", type=int, default=1, help="point num")
    
    args = parser.parse_args()
    return args


def to_device(batch_input, device):
    device_input = {}
    for key, value in batch_input.items():
        if value is not None:
            if key=='image' or key=='label':
                device_input[key] = value.float().to(device)
            elif type(value) is list or type(value) is torch.Size:
                 device_input[key] = value
            else:
                device_input[key] = value.to(device)
        else:
            device_input[key] = value
    return device_input


def postprocess_masks(low_res_masks, image_size, original_size):
    ori_h, ori_w = original_size
    masks = F.interpolate(
        low_res_masks,
        (image_size, image_size),
        mode="bilinear",
        align_corners=False,
        )
    
    if ori_h < image_size and ori_w < image_size:
        top = torch.div((image_size - ori_h), 2, rounding_mode='trunc')  #(image_size - ori_h) // 2
        left = torch.div((image_size - ori_w), 2, rounding_mode='trunc') #(image_size - ori_w) // 2
        masks = masks[..., top : ori_h + top, left : ori_w + left]
        pad = (top, left)
    else:
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        pad = None 
    return masks, pad


def prompt_and_decoder(batched_input, sam_model, image_embeddings, image_size=256, mask_prompt=False, multimask=True,):

    points = (batched_input["point_coords"], batched_input["point_labels"]) if batched_input["point_coords"] is not None else None
    boxes = batched_input.get("boxes", None)
    mask_inputs = batched_input.get("mask_inputs", None) if mask_prompt else None

    with torch.no_grad():
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks = mask_inputs
        )

        low_res_masks, iou_predictions = sam_model.mask_decoder(
            image_embeddings = image_embeddings,
            image_pe = sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask,
        )
    
    if multimask:
        max_values, max_indexs = torch.max(iou_predictions, dim=1)
        max_values = max_values.unsqueeze(1)
        iou_predictions = max_values
        low_res = []
        for i, idx in enumerate(max_indexs):
            low_res.append(low_res_masks[i:i+1, idx])
        low_res_masks = torch.stack(low_res, 0)
    
    masks = F.interpolate(low_res_masks, (image_size, image_size), mode="bilinear", align_corners=False,)
    return masks, low_res_masks, iou_predictions


def fssp_main(sam_model, lr_model, image_embeddings, image_size):
    feat_size = int(image_size/16)
    img_emb = image_embeddings.cpu().numpy().transpose((2, 3, 1, 0)).reshape((feat_size, feat_size, 256)).reshape(-1, 256)
    
    # get the mask predicted by the linear classifier
    y_pred = lr_model.predict(img_emb)
    y_pred = y_pred.reshape((feat_size, feat_size)) 
    # mask predicted by the linear classifier
    mask_pred_l = cv2.resize(y_pred, (image_size, image_size), interpolation=cv2.INTER_NEAREST)

    # use distance transform to find a point inside the mask
    fg_point = get_max_dist_point(mask_pred_l.astype('uint8'))
    # Define the kernel for dilation
    kernel = np.ones((5, 5), np.uint8)
    eroded_mask = cv2.erode(mask_pred_l, kernel, iterations=3)
    mask_pred_l = cv2.dilate(eroded_mask, kernel, iterations=5)
    
    # prompt the sam with the point
    input_point = np.array([[fg_point[0], fg_point[1]]])
    input_label = np.array([1])
    points = (torch.as_tensor(input_point, dtype=torch.float).unsqueeze(0).cuda(), torch.as_tensor(input_label, dtype=torch.int).unsqueeze(0).cuda())
    
    # prompt the sam with the bounding box
    y_indices, x_indices = np.where(mask_pred_l > 0)
    if np.all(mask_pred_l == 0):
        bbox = np.array([0, 0, image_size, image_size])
    else:
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        H, W = mask_pred_l.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
        bbox = np.array([x_min, y_min, x_max, y_max])
    boxes = torch.as_tensor(bbox, dtype=torch.float).cuda()    

    with torch.no_grad():
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=None
        )
        
        masks_pred_sam_prompted, iou_predictions = sam_model.mask_decoder(
            image_embeddings = image_embeddings,
            image_pe = sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
    
    return masks_pred_sam_prompted, iou_predictions, points, boxes


def main(args):
    print("======> Set Parameters for Testing" )
    device = args.device
    run_name = args.run_name
    sam_mode = args.sam_mode
    model_type = args.model_type
    prompt = args.prompt
    strategy = args.strategy
    iter_point = args.iter_point
    point_num = args.point_num
    mask_prompt = args.mask_prompt
    image_size = args.image_size
    multimask = args.multimask
    visual_pred = args.visual_pred
    visual_prompt = args.visual_prompt
    
    # set seed for reproducibility
    seed = 2024   
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    
    print("======> Set Saving Directories and Logs")
    task = args.task
    dataset_name = args.dataset
    root_path = os.path.join(args.root_path, task)
    time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if prompt == 'point':
        save = f'{strategy}_{iter_point}_{time}'
    elif prompt == 'box':
        save = f'{strategy}_{time}'
    elif prompt == 'fssp':
        scale = args.scale
        save = f'{strategy}_{int(scale*100)}_{time}'
    else:
        print('Please check you prompt type!')
        return 0
    save_path = os.path.join(args.save_path, run_name, dataset_name, f'{sam_mode}_{model_type}', prompt, save)
    save_pred_path = os.path.join(save_path, 'pred')
    os.makedirs(save_pred_path, exist_ok=True)

    prompt_logs = os.path.join(save_path, f"{save}_prompts.json")
    loggers = get_logger(os.path.join(save_path, f"{save}.log"))
    loggers.info(f'Args: {args}')
    
    print("======> Load Dataset-Specific Parameters" )
    data_path = os.path.join(root_path, dataset_name)
    batch_size = args.batch_size
    num_workers = args.num_workers
    
    dataset = TestingDataset(data_path=data_path, mode='test', sam_mode=sam_mode, strategy=strategy, point_num=point_num, image_size=image_size, max_pixel=0)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, )
    
    print("======> Set Model")
    sam_checkpoint = args.sam_checkpoint
    encoder_adapter = args.encoder_adapter
    if sam_mode in ['sam', 'med_sam']:
        model = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device) 
    elif sam_mode == 'sam_med2d':
        model = sam_med2d_registry[model_type](image_size=256, sam_checkpoint=sam_checkpoint, encoder_adapter=encoder_adapter).to(device) 
    model.eval()
    criterion = FocalDiceloss_IoULoss()
    metrics = args.metrics
    
    print("======> Start Testing")
    tbar = tqdm((dataloader), total = len(dataloader), leave=False)
    l = len(dataloader)

    prompt_dict = {}
    test_loss = []
    test_metrics = {}
    test_iter_metrics = [0] * len(metrics)
    if dataset_name == 'bhx_sammed':
        test_obj_metrics = {key: [[] for _ in range(len(metrics))] for key in ['right', 'left', 'third', 'fourth']}
    elif dataset_name == 'sabs_sammed':
        test_obj_metrics = {key: [[] for _ in range(len(metrics))] for key in ['spleen', 'rkid', 'lkid', 'gall', 'liver', 'sto', 'aorta', 'pancreas']}

    if prompt == 'fssp':
        lr_models = train_lr_model(model, data_path, dataset_name, image_size, scale=scale)

    for i, batched_input in enumerate(tbar):
        batched_input = to_device(batched_input, device)
        images = batched_input["image"]
        labels = batched_input["label"]
        ori_labels = batched_input["ori_label"]
        original_size = batched_input["original_size"]
        im_name = batched_input['im_name'][0]
        obj = im_name.split('.')[0].split('_')[-1]
        prompt_dict[im_name] = {
            "boxes": batched_input["boxes"].squeeze(1).cpu().numpy().tolist(),
            "point_coords": batched_input["point_coords"].squeeze(1).cpu().numpy().tolist(),
            "point_labels": batched_input["point_labels"].squeeze(1).cpu().numpy().tolist()
            }

        with torch.no_grad():
            image_embeddings = model.image_encoder(images)

        if prompt == 'point':
            batched_input["boxes"] = None
            point_coords, point_labels = [batched_input["point_coords"]], [batched_input["point_labels"]]
     
            for iter in range(iter_point):
                masks, low_res_masks, iou_predictions = prompt_and_decoder(batched_input, model, image_embeddings, image_size, mask_prompt, multimask)
                if iter != iter_point-1:
                    batched_input = generate_point(masks, labels, low_res_masks, batched_input, strategy, point_num, image_size)
                    batched_input = to_device(batched_input, device)
                    point_coords.append(batched_input["point_coords"])
                    point_labels.append(batched_input["point_labels"])
                    batched_input["point_coords"] = torch.concat(point_coords,dim=1)
                    batched_input["point_labels"] = torch.concat(point_labels, dim=1)
  
            points_show = (torch.concat(point_coords, dim=1), torch.concat(point_labels, dim=1))
            boxes_show = None

        elif prompt == 'box':
            batched_input["point_coords"], batched_input["point_labels"] = None, None
            masks, low_res_masks, iou_predictions = prompt_and_decoder(batched_input, model, image_embeddings, image_size, mask_prompt, multimask)
            
            boxes_show = batched_input['boxes']
            points_show = None
            
        elif prompt == 'fssp':
            low_res_masks, iou_predictions, points_show, boxes_show = fssp_main(model, lr_models[obj], image_embeddings, image_size)

        masks, pad = postprocess_masks(low_res_masks, image_size, original_size)
        if visual_pred:
            save_masks(save_pred_path, im_name, masks, ori_labels, image_size, original_size, pad, boxes_show, points_show, visual_prompt=visual_prompt)

        loss = criterion(masks, ori_labels, iou_predictions)
        test_loss.append(loss.item())

        test_batch_metrics = seg_metrics(masks, ori_labels, metrics)
        test_batch_metrics = [float('{:.4f}'.format(metric)) for metric in test_batch_metrics]

        for j in range(len(metrics)):
            test_iter_metrics[j] += test_batch_metrics[j]
            test_obj_metrics[obj][j].append(test_batch_metrics[j])
  
    test_iter_metrics = [metric / l for metric in test_iter_metrics]
    test_metrics = {metrics[i]: '{:.2f}'.format(test_iter_metrics[i] * 100) for i in range(len(test_iter_metrics))}

    for obj in test_obj_metrics.keys():
        for m in range(len(metrics)):
            test_obj_metrics[obj][m] = np.mean(np.array(test_obj_metrics[obj][m]))
        test_obj_metrics[obj] = {metrics[i]: '{:.2f}'.format(test_obj_metrics[obj][i] * 100) for i in range(len(test_obj_metrics[obj]))}

    average_loss = np.mean(test_loss)
    loggers.info(f"Test loss: {average_loss:.4f}")
    loggers.info(f"Test metrics: {test_metrics}")
    loggers.info(f"Test class metrics: {test_obj_metrics}")
    with open(prompt_logs, 'w') as f:
        json.dump(prompt_dict, f, indent=2)


if __name__ == '__main__':
    args = parse_args()
    main(args)
