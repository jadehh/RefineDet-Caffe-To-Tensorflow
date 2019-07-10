from tensorflow.contrib import slim
from datasets import dataset_factory
import tensorflow as tf
from nets.refinedet_vgg_512 import refindet_class
from preprocessing import preprocessing_factory
import numpy as np
from utils import tf_utils
from utils import tfrecord_voc_utils as voc_utils
import os
from jade import *
lr = 0.001
buffer_size = 1024
epochs = 300
reduce_lr_epoch = []
DATA_FORMAT = "NCHW"
# =========================================================================== #
# General Flags.
# =========================================================================== #
train_dir = GetRootPath() + "Models/RefinedetModels/"
CreateSavePath(train_dir)
tf.app.flags.DEFINE_string(
    'train_dir',  train_dir,
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer(
    'batch_size', 16, 'The number of samples in each batch.')
FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.DEBUG)

# Pre-processing image, labels and bboxes.


DATA_FORMAT = 'NHWC'

dataset = dataset_factory.get_dataset(
   "pascalvoc_2012", "train", "/home/jade/Data/VOCdevkit/VOC2012/TFRecords")

image_preprocessing_fn = preprocessing_factory.get_preprocessing(
    "ssd_512_vgg", is_training=True)
with tf.device('/device:CPU:0'):
    provider = slim.dataset_data_provider.DatasetDataProvider(
    dataset,
    num_readers=4,
    common_queue_capacity=20 * 4,
    common_queue_min=10 * 4,
    shuffle=True)

    # Get for SSD network: image, labels, bboxes.
    [image, shape, glabels, gbboxes] = provider.get(['image', 'shape',
                                                 'object/label',
                                                 'object/bbox'])

    image, ground_th = \
        image_preprocessing_fn(image, glabels, gbboxes,
                           out_shape=(512,512),
                           data_format=DATA_FORMAT)
    ground_th = tf.reshape(ground_th,[60,5])



    train_batches = tf.train.batch(
    tf_utils.reshape_list([image,ground_th]),
    batch_size=FLAGS.batch_size,
    num_threads=4,
    capacity=5 * FLAGS.batch_size)


refinedet_net = refindet_class(batch_size=FLAGS.batch_size,train_data=train_batches)
refinedet_anchors = refinedet_net.priors
refinedet_shape = refinedet_net.shape
for i in range(10000):
    print('-'*25, 'epoch', i, '-'*25)
    if i in reduce_lr_epoch:
        lr = lr/10.
        print('reduce lr, lr=', lr, 'now')
    #mean_loss = refinedet_net.train_one_epoch(lr,i,train_batches)
    refinedet_net.train_one_epoch(lr,i)
    #print('>> mean loss', mean_loss)
    refinedet_net.save_weight('latest', FLAGS.train_dir)


