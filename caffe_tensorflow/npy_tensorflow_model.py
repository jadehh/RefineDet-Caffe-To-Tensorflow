from jade.jade_tools import *
import argparse
import tensorflow as tf
from caffe_tensorflow.RefinedetModelNpy  import Refinedet512Model
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--model_path', type=str, default="HAND_GESTURErefindet_512.npy")
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--label_map_path', type=str, default="/home/jade/label_map/hand_gesture_label_map.pbtxt")
args = parser.parse_args()
refinedetModel = Refinedet512Model(args)
saver=tf.train.Saver()
refinedetModel.net.save_ckpt(saver, refinedetModel.sess, "tensorflow_model","HANDGesture")
