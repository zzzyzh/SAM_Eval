CUDA_VISIBLE_DEVICES=3 python test.py --sam_mode sam --model_type vit_b --sam_checkpoint ../../data/experiments/weights/sam_vit_b_01ec64.pth  --prompt box --strategy base
CUDA_VISIBLE_DEVICES=3 python test.py --sam_mode sam --model_type vit_h --sam_checkpoint ../../data/experiments/weights/sam_vit_h_4b8939.pth  --prompt box --strategy base
CUDA_VISIBLE_DEVICES=3 python test.py --sam_mode sam_med2d --model_type vit_b --sam_checkpoint ../../data/experiments/weights/sam-med2d_b.pth --image_size 256 --prompt box --strategy base
CUDA_VISIBLE_DEVICES=3 python test.py --sam_mode med_sam --model_type vit_b --sam_checkpoint ../../data/experiments/weights/medsam_vit_b.pth  --prompt box --strategy base

CUDA_VISIBLE_DEVICES=3 python test.py --sam_mode sam --model_type vit_b --sam_checkpoint ../../data/experiments/weights/sam_vit_b_01ec64.pth  --prompt point --strategy base --iter_point 1
CUDA_VISIBLE_DEVICES=3 python test.py --sam_mode sam --model_type vit_b --sam_checkpoint ../../data/experiments/weights/sam_vit_b_01ec64.pth  --prompt point --strategy base --iter_point 5
CUDA_VISIBLE_DEVICES=3 python test.py --sam_mode sam --model_type vit_b --sam_checkpoint ../../data/experiments/weights/sam_vit_b_01ec64.pth  --prompt point --strategy base --iter_point 10

CUDA_VISIBLE_DEVICES=3 python test.py --sam_mode sam --model_type vit_h --sam_checkpoint ../../data/experiments/weights/sam_vit_h_4b8939.pth  --prompt point --strategy base --iter_point 1
CUDA_VISIBLE_DEVICES=3 python test.py --sam_mode sam --model_type vit_h --sam_checkpoint ../../data/experiments/weights/sam_vit_h_4b8939.pth  --prompt point --strategy base --iter_point 5
CUDA_VISIBLE_DEVICES=3 python test.py --sam_mode sam --model_type vit_h --sam_checkpoint ../../data/experiments/weights/sam_vit_h_4b8939.pth  --prompt point --strategy base --iter_point 10

CUDA_VISIBLE_DEVICES=3 python test.py --sam_mode sam_med2d --model_type vit_b --sam_checkpoint ../../data/experiments/weights/sam-med2d_b.pth --image_size 256 --prompt point --strategy base --iter_point 1 --mask_prompt True
CUDA_VISIBLE_DEVICES=3 python test.py --sam_mode sam_med2d --model_type vit_b --sam_checkpoint ../../data/experiments/weights/sam-med2d_b.pth --image_size 256 --prompt point --strategy base --iter_point 5 --mask_prompt True
CUDA_VISIBLE_DEVICES=3 python test.py --sam_mode sam_med2d --model_type vit_b --sam_checkpoint ../../data/experiments/weights/sam-med2d_b.pth --image_size 256 --prompt point --strategy base --iter_point 10 --mask_prompt True

CUDA_VISIBLE_DEVICES=3 python test.py --sam_mode med_sam --model_type vit_b --sam_checkpoint ../../data/experiments/weights/medsam_vit_b.pth  --prompt point --strategy base --iter_point 1 --mask_prompt True
CUDA_VISIBLE_DEVICES=3 python test.py --sam_mode med_sam --model_type vit_b --sam_checkpoint ../../data/experiments/weights/medsam_vit_b.pth  --prompt point --strategy base --iter_point 5 --mask_prompt True
CUDA_VISIBLE_DEVICES=3 python test.py --sam_mode med_sam --model_type vit_b --sam_checkpoint ../../data/experiments/weights/medsam_vit_b.pth  --prompt point --strategy base --iter_point 10 --mask_prompt True
