# sam-vit_b
# base
# CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam --model_type vit_b --sam_checkpoint /home/yanzhonghao/data/ven/weights/sam_vit_b_01ec64.pth --image_size 1024 --prompt point --strategy base --iter_point 1
# CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam --model_type vit_b --sam_checkpoint /home/yanzhonghao/data/ven/weights/sam_vit_b_01ec64.pth --image_size 1024 --prompt point --strategy base --iter_point 5
# CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam --model_type vit_b --sam_checkpoint /home/yanzhonghao/data/ven/weights/sam_vit_b_01ec64.pth --image_size 1024 --prompt point --strategy base --iter_point 10

# far
# CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam --model_type vit_b --sam_checkpoint /home/yanzhonghao/data/ven/weights/sam_vit_b_01ec64.pth --image_size 1024 --prompt point --strategy far --iter_point 1
# CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam --model_type vit_b --sam_checkpoint /home/yanzhonghao/data/ven/weights/sam_vit_b_01ec64.pth --image_size 1024 --prompt point --strategy far --iter_point 5
# CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam --model_type vit_b --sam_checkpoint /home/yanzhonghao/data/ven/weights/sam_vit_b_01ec64.pth --image_size 1024 --prompt point --strategy far --iter_point 10

#############################

# sam-vit_h
# base
CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam --model_type vit_h --sam_checkpoint /home/yanzhonghao/data/ven/weights/sam_vit_h_4b8939.pth --image_size 1024 --prompt point --strategy base --iter_point 1
CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam --model_type vit_h --sam_checkpoint /home/yanzhonghao/data/ven/weights/sam_vit_h_4b8939.pth --image_size 1024 --prompt point --strategy base --iter_point 5
CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam --model_type vit_h --sam_checkpoint /home/yanzhonghao/data/ven/weights/sam_vit_h_4b8939.pth --image_size 1024 --prompt point --strategy base --iter_point 10

# far
CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam --model_type vit_h --sam_checkpoint /home/yanzhonghao/data/ven/weights/sam_vit_h_4b8939.pth --image_size 1024 --prompt point --strategy far --iter_point 1
CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam --model_type vit_h --sam_checkpoint /home/yanzhonghao/data/ven/weights/sam_vit_h_4b8939.pth --image_size 1024 --prompt point --strategy far --iter_point 5
CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam --model_type vit_h --sam_checkpoint /home/yanzhonghao/data/ven/weights/sam_vit_h_4b8939.pth --image_size 1024 --prompt point --strategy far --iter_point 10

#############################

# sam_med2d
# base
# CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam_med2d --model_type vit_b --sam_checkpoint /home/yanzhonghao/data/ven/sam-med2d/sam-med2d_b.pth --prompt point --strategy base --iter_point 1
# CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam_med2d --model_type vit_b --sam_checkpoint /home/yanzhonghao/data/ven/sam-med2d/sam-med2d_b.pth --prompt point --strategy base --iter_point 5
# CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam_med2d --model_type vit_b --sam_checkpoint /home/yanzhonghao/data/ven/sam-med2d/sam-med2d_b.pth --prompt point --strategy base --iter_point 10

# far
# CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam_med2d --model_type vit_b --sam_checkpoint /home/yanzhonghao/data/ven/sam-med2d/sam-med2d_b.pth --prompt point --strategy far --iter_point 1
# CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam_med2d --model_type vit_b --sam_checkpoint /home/yanzhonghao/data/ven/sam-med2d/sam-med2d_b.pth --prompt point --strategy far --iter_point 5
# CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam_med2d --model_type vit_b --sam_checkpoint /home/yanzhonghao/data/ven/sam-med2d/sam-med2d_b.pth --prompt point --strategy far --iter_point 10
