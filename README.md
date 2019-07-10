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

3.clean the dataset rewrite the train.txt and the test.txt the train.txt looks like
> VOC2012/JPEGImages/2008_000340.jpg VOC2012/Annotations/2008_000340.xml
```
python create_dataset 
```

4.prepare dataset
```
python create_lmdb.py 
```

5.use train512 train a 512 Model
```
python train512
```
6.caffe_to_tensorflow.py caffeModel to npy
```
python caffe_to_tensorflow 
```
> use the Caffe Trained Model to save as npy

7.npy to tensorflow Model
```
python caffe_tensorflow/npy_tensorflow_model.py 
```
>change .npy to tensorflow Model