# SibNet: Food Instance Counting and Segmentation
This is an implementation of SibNet, a deep learning model for Instance Counting and Segmentation. In this project, we train SibNet on three food datasets including Cookie, Dimsum, Sushi representing Western, Chinese and Japanese food. The images and segmentation masks are publicly available to download [here](https://drive.google.com/file/d/1tXtxZE7cI1uhbay_b6I4qEz86zh0R8kq/view?usp=sharing).
![SibNet Performance](images/sibnet_result.png)

## Source code
The source code includes:
* The architecture of SibNet
* The class to train, test and deloy SibNet
* The framework for training and testing SibNet on three food datasets

## Installation
1. Setup and activate anaconda environment:
    ```bash
    conda env update --file environment.yml --prune
    source ~/.bashrc
    conda activate sibnetenv
    ```
1. Setup cython libraries: direct to src/alcython/ and run following bash command
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


