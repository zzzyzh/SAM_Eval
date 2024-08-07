# sam-vit_b
# base
CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam --model_type vit_b --sam_checkpoint ../../data/experiments/weights/sam_vit_b_01ec64.pth  --prompt point --strategy base --iter_point 1
CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam --model_type vit_b --sam_checkpoint ../../data/experiments/weights/sam_vit_b_01ec64.pth  --prompt point --strategy base --iter_point 5
CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam --model_type vit_b --sam_checkpoint ../../data/experiments/weights/sam_vit_b_01ec64.pth  --prompt point --strategy base --iter_point 10

# far
# CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam --model_type vit_b --sam_checkpoint ../../data/experiments/weights/sam_vit_b_01ec64.pth  --prompt point --strategy far --iter_point 1
# CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam --model_type vit_b --sam_checkpoint ../../data/experiments/weights/sam_vit_b_01ec64.pth  --prompt point --strategy far --iter_point 5
# CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam --model_type vit_b --sam_checkpoint ../../data/experiments/weights/sam_vit_b_01ec64.pth  --prompt point --strategy far --iter_point 10

# m_area
# CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam --model_type vit_b --sam_checkpoint ../../data/experiments/weights/sam_vit_b_01ec64.pth  --prompt point --strategy m_area --iter_point 1
# CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam --model_type vit_b --sam_checkpoint ../../data/experiments/weights/sam_vit_b_01ec64.pth  --prompt point --strategy m_area --iter_point 5
# CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam --model_type vit_b --sam_checkpoint ../../data/experiments/weights/sam_vit_b_01ec64.pth  --prompt point --strategy m_area --iter_point 10

#############################

# sam-vit_h
# base
CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam --model_type vit_h --sam_checkpoint ../../data/experiments/weights/sam_vit_h_4b8939.pth  --prompt point --strategy base --iter_point 1
CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam --model_type vit_h --sam_checkpoint ../../data/experiments/weights/sam_vit_h_4b8939.pth  --prompt point --strategy base --iter_point 5
CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam --model_type vit_h --sam_checkpoint ../../data/experiments/weights/sam_vit_h_4b8939.pth  --prompt point --strategy base --iter_point 10

# far
# CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam --model_type vit_h --sam_checkpoint ../../data/experiments/weights/sam_vit_h_4b8939.pth  --prompt point --strategy far --iter_point 1
# CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam --model_type vit_h --sam_checkpoint ../../data/experiments/weights/sam_vit_h_4b8939.pth  --prompt point --strategy far --iter_point 5
# CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam --model_type vit_h --sam_checkpoint ../../data/experiments/weights/sam_vit_h_4b8939.pth  --prompt point --strategy far --iter_point 10

# m_area
# CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam --model_type vit_h --sam_checkpoint ../../data/experiments/weights/sam_vit_h_4b8939.pth  --prompt point --strategy m_area --iter_point 1
# CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam --model_type vit_h --sam_checkpoint ../../data/experiments/weights/sam_vit_h_4b8939.pth  --prompt point --strategy m_area --iter_point 5
# CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam --model_type vit_h --sam_checkpoint ../../data/experiments/weights/sam_vit_h_4b8939.pth  --prompt point --strategy m_area --iter_point 10

#############################

# sam_med2d
# base
CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam_med2d --model_type vit_b --sam_checkpoint ../../data/experiments/weights/sam-med2d_b.pth --image_size 256 --prompt point --strategy base --iter_point 1 --mask_prompt True
CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam_med2d --model_type vit_b --sam_checkpoint ../../data/experiments/weights/sam-med2d_b.pth --image_size 256 --prompt point --strategy base --iter_point 5 --mask_prompt True
CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam_med2d --model_type vit_b --sam_checkpoint ../../data/experiments/weights/sam-med2d_b.pth --image_size 256 --prompt point --strategy base --iter_point 10 --mask_prompt True

# far
# CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam_med2d --model_type vit_b --sam_checkpoint ../../data/experiments/weights/sam-med2d_b.pth --image_size 256 --prompt point --strategy far --iter_point 1 --mask_prompt True
# CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam_med2d --model_type vit_b --sam_checkpoint ../../data/experiments/weights/sam-med2d_b.pth --image_size 256 --prompt point --strategy far --iter_point 5 --mask_prompt True
# CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam_med2d --model_type vit_b --sam_checkpoint ../../data/experiments/weights/sam-med2d_b.pth --image_size 256 --prompt point --strategy far --iter_point 10 --mask_prompt True

# m_area
# CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam_med2d --model_type vit_b --sam_checkpoint ../../data/experiments/weights/sam-med2d_b.pth --image_size 256 --prompt point --strategy m_area --iter_point 1 --mask_prompt True
# CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam_med2d --model_type vit_b --sam_checkpoint ../../data/experiments/weights/sam-med2d_b.pth --image_size 256 --prompt point --strategy m_area --iter_point 5 --mask_prompt True
# CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam_med2d --model_type vit_b --sam_checkpoint ../../data/experiments/weights/sam-med2d_b.pth --image_size 256 --prompt point --strategy m_area --iter_point 10 --mask_prompt True

#############################

# med_sam
# base
CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode med_sam --model_type vit_b --sam_checkpoint ../../data/experiments/weights/medsam_vit_b.pth  --prompt point --strategy base --iter_point 1 --mask_prompt True
CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode med_sam --model_type vit_b --sam_checkpoint ../../data/experiments/weights/medsam_vit_b.pth  --prompt point --strategy base --iter_point 5 --mask_prompt True
CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode med_sam --model_type vit_b --sam_checkpoint ../../data/experiments/weights/medsam_vit_b.pth  --prompt point --strategy base --iter_point 10 --mask_prompt True

# far
# CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode med_sam --model_type vit_b --sam_checkpoint ../../data/experiments/weights/medsam_vit_b.pth  --prompt point --strategy far --iter_point 1 --mask_prompt True
# CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode med_sam --model_type vit_b --sam_checkpoint ../../data/experiments/weights/medsam_vit_b.pth  --prompt point --strategy far --iter_point 5 --mask_prompt True
# CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode med_sam --model_type vit_b --sam_checkpoint ../../data/experiments/weights/medsam_vit_b.pth  --prompt point --strategy far --iter_point 10 --mask_prompt True

# m_area
# CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode med_sam --model_type vit_b --sam_checkpoint ../../data/experiments/weights/medsam_vit_b.pth  --prompt point --strategy m_area --iter_point 1 --mask_prompt True
# CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode med_sam --model_type vit_b --sam_checkpoint ../../data/experiments/weights/medsam_vit_b.pth  --prompt point --strategy m_area --iter_point 5 --mask_prompt True
# CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode med_sam --model_type vit_b --sam_checkpoint ../../data/experiments/weights/medsam_vit_b.pth  --prompt point --strategy m_area --iter_point 10 --mask_prompt True
