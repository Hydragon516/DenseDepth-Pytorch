# DenseDepth-Pytorch

This repository is an unofficial Pytorch implementation of ["High Quality Monocular Depth Estimation via Transfer Learning"](https://arxiv.org/abs/1812.11941). 
The official implementation can be found [here](https://github.com/ialhashim/DenseDepth).

## Dataset

[NYU Depth Dataset V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) dataset is required. 
You can download the preprocessed dataset [here](https://drive.google.com/u/0/uc?export=download&confirm=QLMA&id=1fdFu5NGXe4rTLYKD5wOqk9dl-eJOefXo).
You do not need to unzip this file.

## Training

* Open the [configs.py](https://github.com/Hydragon516/DenseDepth-Pytorch/blob/master/configs/configs.py) file and edit the dataset path and training options.

* For training, use the following simple command.

```shell
$ python train_nyuv2.py
```

## TODO

- [ ] Results for NYU Depth Dataset V2
- [ ] complete the KITTI data loader
- [ ] Visualization code
