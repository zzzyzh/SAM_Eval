# SAM_Eval

## Introduction
SAM Eval is a comprehensive evaluation framework designed to facilitate prompt selection strategies for segmentation tasks using SAM, SAM2, and a variety of related models. This project focuses on systematically assessing the performance of SAM-like models when directly applied to medical imaging scenarios. Naturally, the methods and strategies proposed in SAM Eval are also applicable to natural image segmentation tasks.

Built upon the [SAM-Med2D](https://github.com/OpenGVLab/SAM-Med2D) &#8194; foundation, SAM Eval incorporates rigorous evaluation using the AbdomenAtlas1.1Mini [AbdomenAtlas1.1Mini](https://huggingface.co/datasets/AbdomenAtlas/_AbdomenAtlas1.1Mini) &#8194; dataset, which includes detailed annotations for multiple abdominal organs. This enables in-depth analysis of model behavior and prompt effectiveness in realistic clinical settings.

We invite you to explore and use SAM Eval in your own projects—whether in the medical domain or beyond!

## Updates
- (2025.05.07) Pre-process code of AbdomenAtlas1.1 release
- (2024.04.09) Inference code of MedSAM release
- (2024.04.06) Inference code of SAM and SAM-Med2D release

## Installation
Following [Segment Anything](https://github.com/facebookresearch/segment-anything), the code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. 

For SAM_Eval, `python=3.10` and `pytorch=2.3.1` are used.

1. Clone the repository.
      ```
      git clone https://github.com/zzzyzh/SAM_Eval.git
      cd SAM_Eval
      ```
2. Create a virtual environment for SAM_Eval and activate the environment.
    ```
    conda create -n sam_eval python=3.10 -y
    conda activate sam_eval
    ```
3. Install Pytorch and TorchVision. 
    ```
    pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
    ```
4. Install other dependencies.
    ```
    pip install -r requirements.txt
    ```

## Contribution
Our project primarily evaluates the segmentation effects of different prompting methods. Users can set the strategy parameter to explore how various strategies impact the segmentation outcomes.

- point
    - base: Identify areas of incorrect prediction and randomly select points from these mispredicted areas.
    - far: Identify areas of incorrect prediction and choose the point furthest from the boundary within these mispredicted areas.
    - m_area: Identify areas of incorrect prediction, divide them into false positive and false negative areas, and choose a point furthest from the boundary in the area with the larger surface area.
- box
    - base: The bounding rectangle of the mask.
    - square_max: Construct a square using the longer side of the mask's bounding rectangle, with the center coinciding with the mask's centroid.
    - square_min: Construct a square using the shorter side of the mask's bounding rectangle, with the center coinciding with the mask's centroid.

## Data Preparation

### Download
1. **AbdomenAtlas1.1**  [AbdomenAtlas1.1](https://huggingface.co/datasets/AbdomenAtlas/_AbdomenAtlas1.1Mini)

### Pre-processing
Please refer to [Cheng et al.](https://arxiv.org/abs/2308.16184)

Our data follows the data preprocessing process mentioned in the [SAM-Med2D](https://github.com/OpenGVLab/SAM-Med2D) &#8194; paper. You can refer to `dataset_split/prepare_dataset.py` to process your own dataset.

    ```
    python dataset_split/prepare_dataset.py
    ```

## Test
- point
    ```
    CUDA_VISIBLE_DEVICES=0 python eval_sam.py --sam_mode sam --dataset_name AbdomenAtlas1.1 --image_size 1024 --model_type vit_b --prompt point --strategy base --iter_point 5
    ```
- box
    ```
    CUDA_VISIBLE_DEVICES=0 python eval_sam.py --sam_mode sam --dataset_name AbdomenAtlas1.1 --image_size 1024 --model_type vit_b --prompt box --strategy base
    ```

## Generate a prompt list for the dataset locally
Considering that most existing evaluation methods require prompts to be generated based on masks, for images without masks, the only way to utilize SAM to produce segmentation results is by manually selecting points and boxes on the image, recording their coordinates, and then submitting them to SAM to generate the segmentation outcome. For this reason, I have written a script that can generate a prompt list for a set of images locally in advance. Everyone can download `find_prompt_gui.py` to use locally.
- It features two modes (point and box), which can be switched by clicking `m`.
    - In point mode, clicking any location on the image will display the coordinates of that point on the image and record these coordinates.
    - In box mode, drawing a box around any object on the image will display the corresponding box and record the coordinates of the box's top-left and bottom-right corners.
- A `.json` file will be generated for each group of images, recording the coordinates of the prompts for each image.
- Clicking `q` will change the image.
- Clicking `esc` will exit the program.
- Due to certain reasons, after you run the program, you need to click `m` once to enter the box mode at first.
- example:
<p align="center"><img width="400" alt="image" src="./pic/find_prompt_example.png"></p> 

## Citation
```bash
@misc{SAM_Eval_2023,
  author = {Yan, Zhonghao},
  title = {{Segment Anything Model for Medical Imgae Evaluation}},
  year = {2023},
  howpublished = {\url{https://github.com/zzzyzh/SAM_Eval}},
}
```

## Acknowledgement
Thanks to the open-source of the following projects
- [Segment Anything](https://github.com/facebookresearch/segment-anything) &#8194; 
- [SAM-Med2D](https://github.com/OpenGVLab/SAM-Med2D) &#8194;

## Contact Us
If you have any suggestions for the project, or would like to discuss further expansions with us, please contact us at the following email: zhonghao.yan@bupt.edu.cn or yanzhonghao531@gmail.com. We look forward to your valuable input!