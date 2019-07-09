# RefineDet-Caffe-To-Tensorflow

Refinedet 网络 Caffe模型 转 Tensorflow模型

1.先安装Refinedet Caffe 版本
> [Refinedet Caffe](https://github.com/sfzhang15/RefineDet)

2.Installation
- Install [PyTorch-0.2.0-0.3.1](http://pytorch.org/) by selecting your environment on the website and running the appropriate command.
- Clone this repository. This repository is mainly based on[RFBNet](https://github.com/ruinmessi/RFBNet), [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch) and [Chainer-ssd](https://github.com/Hakuyume/chainer-ssd), a huge thank to them.
  * Note: We currently only support Python 3+.
- Compile the nms and coco tools:
```Shell
./make.sh
```
Note*: Check you GPU architecture support in utils/build.py, line 131. Default is:
    
3.使用train512.py训练一个模型

4.caffe_to_tensorflow.py caffeModel to npy

5.npy to tensorflow Model
