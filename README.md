# RefineDet-Caffe-To-Tensorflow

Refinedet Caffe To Tensorflow模型

1.Install Refinedet Caffe 
> [Refinedet Caffe](https://github.com/sfzhang15/RefineDet)

2.Installation
- Compile the nms:
```Shell
./make.sh
```
Note*: Check you GPU architecture support in utils/build.py, line 131. Default is:
3.prepare dataset
```
python create_lmdb.py 
```
4.use train512 train a 512 Model
```
python train512
```
5.caffe_to_tensorflow.py caffeModel to npy

6.npy to tensorflow Model
