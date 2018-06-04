# iResNet

This repository contains the code (in CAFFE) for "[Learning for Disparity Estimation through Feature Constancy](https://arxiv.org/abs/1712.01039)" paper (CVPR 2018) by Zhengfa Liang.
 


### Citation
```
@article{Liang2018Learning,
  title={Learning for Disparity Estimation through Feature Constancy},
  author={Liang, Zhengfa and Feng, Yiliu and Guo, Yulan and Liu, Hengzhu and Chen, Wei and Qiao, Linbo and Zhou, Li and Zhang, Jianfeng},
  booktitle={Computer Vision and Pattern Recognition},
  year={2018},
}
```

## Contents

1. [Usage](#usage)
2. [Contacts](#contact)

## Usage

### Dependencies
* Ubuntu 16.04
* [Python2.7](https://www.python.org/downloads/)
* Caffe
* CUDNN 5.1
* CUDA 8.0
* [Scene Flow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
* [ETH3D2017](https://www.eth3d.net/datasets)
* [Kitti2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)
* [Middlebury2014](http://vision.middlebury.edu/stereo/data/)

Notes: 

- You should first install Caffe following the [Installation instructions](http://caffe.berkeleyvision.org/installation.html) here. 

- The caffe code in this repository is modiffied from [DispNet](https://lmb.informatik.uni-freiburg.de/resources/software.php), which includes the "Correlation1D" layer.

- The FlowWarp layer is from [FlowNet 2.0](https://github.com/lmb-freiburg/flownet2)

- We add RandomCrop layer and DataSwitch layer.

- RandomCrop is used to crop bottom blob to desired width and height, but channel number of this layer is fixed to 7 (left image, right image, and disparity). If the desired width or height is larger than that of bottom blob, we use 128 to fill the first 6 channels, and use NaN to fill the last channels.

```
layer {  name: "Random_crop_kitti2015"
  type: "RandomCrop"
  bottom: "kitti2015_data"
  top: "kitti2015_cropped_data"
  random_crop_param { target_height: 350  target_width: 694}
}
```

- DataSwitch is used to randomly select one of the input bottom blobs as output.
```
layer {  name: "Random_select_datasets"
  type: "DataSwitch"
  bottom: "MiddleBury_cropped_data"
  bottom: "kitti2015_cropped_data"
  bottom: "eth3d_cropped_data"
  top: "curr_data"
}
```


### Data preparation

Download datasets using the instructions from http://www.cvlibs.net:3000/ageiger/rob_devkit. Put the folder "datasets_middlebury2014" under "CAFFE_ROOT/data". The file structure looks like:
```
+── caffe
│   +── data
│       +── datasets_middlebury2014
│           +── metadata
│           +── test
│           +── training
```

### Training

2. Enter folder "CAFFE_ROOT/data", and use MATLAB to run the script "make_lmdbs.m"


3. Enter folder "CAFFE_ROOT/models/model", and run:
```
python ../train_rob.py 2>&1 | tee rob.log
```



### Evaluattion

Download the pretrained model from [Pretrained Model], and place it in the folder CAFFE_ROOT/models/model. You need to modify CAFFE_ROOT at line 15 in file "test_rob.py". The results for submission will be stored at CAFFE_ROOT/models/submission_results.

```
  cd models
  python test_rob.py model/iResNet_ROB.caffemodel
```


### Pretrained Model

CVPR 2018

| Scene Flow( for fine-tuning kitti)|  KITTI 2015 |
|---|---|
|[Baiduyun](https://pan.baidu.com/s/1yzopXEVoon2GTO2z-E9gZA)|[Baiduyun](https://pan.baidu.com/s/1_IPicEoPD-9xey2LoL556Q)|

ROB 2018

| Scene Flow |  Final model |
|---|---|
|[Baiduyun](https://pan.baidu.com/s/1ziHbZc37SVvhkpM0hStJng)|[Baiduyun](https://pan.baidu.com/s/1LZkUb0HHUihEoKgp4vCaTw)|


## Contact
liangzhengfa10@nudt.edu.cn
