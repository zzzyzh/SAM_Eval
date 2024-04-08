# SAM_Eval

## Introduction

Welcome to the official repository for the SAM Inference Project! This repository utilizes the pre-trained weights of SAM to provide a comprehensive platform for inference and evaluation. We are committed to building an easy-to-use and flexible SAM inference environment that meets the diverse needs of users for inference performance and results.

Our repository is based on the [SAM-Med2D](https://github.com/OpenGVLab/SAM-Med2D) &#8194; implementation, if you want to go deeper, please refer to [SAM-Med2D](https://github.com/OpenGVLab/SAM-Med2D) &#8194;!

## Updates
- (2024.04.06) Inference code of SAM and SAM-Med2D release

## Installation
Following [Segment Anything](https://github.com/facebookresearch/segment-anything), the code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. 

For SAM_Eval, `python=3.8`, `pytorch=1.11.0`, and `torchvision=0.12.0` are used.

1. Clone the repository.
      ```
      git clone https://github.com/zzzyzh/SAM_Eval.git
      cd SAM_Eval
      ```
2. Create a virtual environment for FSP-SAM and and activate the environment.
    ```
    conda create -n sam_eval python=3.8
    conda activate sam_eval
    ```
3. Install Pytorch and TorchVision. 
    ```
    conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=10.2 -c pytorch
    ```
    or
    ```
    conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
    ```
4. Install other dependencies.
    ```
    pip install -r requirements.txt
    ```

## Test
- prepare dataset
    - Our data follows the data preprocessing process mentioned in the [SAM-Med2D](https://github.com/OpenGVLab/SAM-Med2D) &#8194; paper. You can refer to `pre_data.py` to process your own dataset.
- point
    ```
    python test.py --sam_mode sam --model_type vit_b --sam_checkpoint ../sam_vit_b_01ec64.pth --image_size 1024 --prompt point --strategy base --iter_point 1
    ```
- box
    ```
    CUDA_VISIBLE_DEVICES=1 python test.py --sam_mode sam_med2d --model_type vit_b --sam_checkpoint ../sam-med2d_b.pth --prompt box --strategy base
    ```

## Generate a prompt list for the dataset locally

Considering that most existing evaluation methods require prompts to be generated based on masks, for images without masks, the only way to utilize SAM to produce segmentation results is by manually selecting points and boxes on the image, recording their coordinates, and then submitting them to SAM to generate the segmentation outcome. For this reason, I have written a script that can generate a prompt list for a set of images locally in advance. Everyone can download `find_prompt_gui.py` to use locally.
- `q`: Quit the application.
- `m`: Switch the mode of selecting prompts (starts with `point`, a single press of `m` switches to `box`).
    - Due to certain reasons, after you run the program, you need to click `m` once to enter the box mode.
- example:
<p align="center"><img width="400" alt="image" src="./pic/find_prompt_example.png"></p> 

## Acknowledgement
Thanks to the open-source of the following projects
- [Segment Anything](https://github.com/facebookresearch/segment-anything) &#8194; 
- [SAM-Med2D](https://github.com/OpenGVLab/SAM-Med2D) &#8194;

## Contact Us
If you have any suggestions for the project, or would like to discuss further expansions with us, please contact us at the following email: zzzyzh@qq.com or zzzyzh@bupt.edu.cn. We look forward to your valuable input!