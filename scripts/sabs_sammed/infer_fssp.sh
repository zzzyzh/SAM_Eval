# sam-vit_b
CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam --task abdomen --dataset sabs_sammed --model_type vit_b --sam_checkpoint ../../data/experiments/weights/sam_vit_b_01ec64.pth --prompt fssp --scale 0.1

CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam --task abdomen --dataset sabs_sammed --model_type vit_b --sam_checkpoint ../../data/experiments/weights/sam_vit_b_01ec64.pth --prompt fssp --scale 0.05

#############################

# sam_med2d
CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam_med2d --task abdomen --dataset sabs_sammed --model_type vit_b --sam_checkpoint ../../data/experiments/weights/sam-med2d_b.pth --image_size 256 --prompt fssp --scale 0.1

CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode sam_med2d --task abdomen --dataset sabs_sammed --model_type vit_b --sam_checkpoint ../../data/experiments/weights/sam-med2d_b.pth --image_size 256 --prompt fssp --scale 0.05

#############################

# med_sam
CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode med_sam --task abdomen --dataset sabs_sammed --model_type vit_b --sam_checkpoint ../../data/experiments/weights/medsam_vit_b.pth  --prompt fssp --scale 0.1

CUDA_VISIBLE_DEVICES=0 python test.py --sam_mode med_sam --task abdomen --dataset sabs_sammed --model_type vit_b --sam_checkpoint ../../data/experiments/weights/medsam_vit_b.pth  --prompt fssp --scale 0.05

