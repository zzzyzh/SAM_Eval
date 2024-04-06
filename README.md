# SAM_Eval

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
4. Install other dependencies.
    ```
    pip install -r requirements.txt
    ```
