# SibNet: Food Instance Counting and Segmentation
Food computing has recently attracted considerable research attention due to its significance for health risk analysis. In the literature, the majority of research efforts are dedicated to food recognition. Relatively few works are conducted for food counting and segmentation, which are essential for portion size estimation. This paper presents a deep neural network, named SibNet, for simultaneous counting and extraction of food instances from an image. The problem is challenging due to varying size and shape of food as well as arbitrary viewing angle of camera, not to mention that food instances often occlude each other. SibNet is novel for proposal of learning seed map to minimize the overlap between instances. The map facilitates counting and can be completed as an instance segmentation map that depicts the arbitrary shape and size of individual instance under occlusion. To this end, a novel sibling relation sub-network is proposed for pixel connectivity analysis. Along with this paper, three new datasets covering Western, Chinese and Japanese food are also constructed for performance evaluation. 

Check out our journal paper [here](https://www.sciencedirect.com/science/article/pii/S0031320321006464).

The images and segmentation masks are publicly available to download [here](https://drive.google.com/file/d/1tXtxZE7cI1uhbay_b6I4qEz86zh0R8kq/view?usp=sharing).

![SibNet Performance](images/sibnet_result.png)

## Source code
The source code includes the implementations for:
* SibNet's architecture
* Evaluation metrics
* Data pre-processing
* Fundamental functions for training, testing, visualizing the results
* Instance extraction algorithms written in Cython

## Installation
1. Setup and activate anaconda environment:
    ```bash
    conda env update --file environment.yml --prune
    source ~/.bashrc
    conda activate sibnetenv
    ```
1. Setup cython libraries: direct to src/alcython/ and run the following bash command
    ```bash
    python setup.py build_ext --inplace
    ```
## Train and test SibNet model
The following configurations and commands works under the **src** folder
1. Config the path to data images, segmentation masks, seed masks and sibling relation maps in *constants.py*
1. How to train:
    * Edit the data type in *config.py*
    * Run ```python main.py train```
1. How to test:
    * Edit the path to trained model (ckpt_path) in *config.py*
    * Run ```python main.py test```

Our weights trained on three food datasets are found [here](https://drive.google.com/drive/folders/1ClsAx27mm3qFFn8EDkQkhOQ4CKtQPp80?usp=sharing).

## Citation
```
@article{NGUYEN2022108470,
title = {SibNet: Food instance counting and segmentation},
journal = {Pattern Recognition},
volume = {124},
pages = {108470},
year = {2022},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2021.108470},
url = {https://www.sciencedirect.com/science/article/pii/S0031320321006464},
author = {Huu-Thanh Nguyen and Chong-Wah Ngo and Wing-Kwong Chan},
}
```

