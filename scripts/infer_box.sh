# sam-vit_b
# base
CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam --model_type vit_b --sam_checkpoint /home/yanzhonghao/data/ven/weights/sam_vit_b_01ec64.pth --image_size 1024 --prompt box --strategy base

CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam --model_type vit_b --sam_checkpoint /home/yanzhonghao/data/ven/weights/sam_vit_b_01ec64.pth --image_size 1024 --prompt box --strategy square_max
CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam --model_type vit_b --sam_checkpoint /home/yanzhonghao/data/ven/weights/sam_vit_b_01ec64.pth --image_size 1024 --prompt box --strategy square_min

#############################

# sam-vit_h
# base
CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam --model_type vit_h --sam_checkpoint /home/yanzhonghao/data/ven/weights/sam_vit_h_4b8939.pth --image_size 1024 --prompt box --strategy base

CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam --model_type vit_h --sam_checkpoint /home/yanzhonghao/data/ven/weights/sam_vit_h_4b8939.pth --image_size 1024 --prompt box --strategy square_max
CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam --model_type vit_h --sam_checkpoint /home/yanzhonghao/data/ven/weights/sam_vit_h_4b8939.pth --image_size 1024 --prompt box --strategy square_min

#############################

# sam_med2d
# base
CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam_med2d --model_type vit_b --sam_checkpoint /home/yanzhonghao/data/ven/weights/sam-med2d_b.pth --prompt box --strategy base

CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam_med2d --model_type vit_b --sam_checkpoint /home/yanzhonghao/data/ven/weights/sam-med2d_b.pth --prompt box --strategy square_max
CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam_med2d --model_type vit_b --sam_checkpoint /home/yanzhonghao/data/ven/weights/sam-med2d_b.pth --prompt box --strategy square_min

#############################

# med_sam
# base
CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode med_sam --model_type vit_b --sam_checkpoint /home/yanzhonghao/data/ven/weights/medsam_vit_b.pth --image_size 1024 --prompt box --strategy base

CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode med_sam --model_type vit_b --sam_checkpoint /home/yanzhonghao/data/ven/weights/medsam_vit_b.pth --image_size 1024 --prompt box --strategy square_max
CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode med_sam --model_type vit_b --sam_checkpoint /home/yanzhonghao/data/ven/weights/medsam_vit_b.pth --image_size 1024 --prompt box --strategy square_min
