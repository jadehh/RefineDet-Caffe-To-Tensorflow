'''
In this example, we will load a RefineDet model and use it to detect objects.
'''

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
# Make sure that caffe is on the python path:
from jade import *

sys.path.append(GetRootPath()+"/RefineDet/python")
import caffe

from google.protobuf import text_format
from caffe.proto import caffe_pb2
import cv2


def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames


def ShowResults(img, results, labelmap, threshold=0.6):
    num_classes = len(labelmap.item) - 1
    bboxes = []
    scores = []
    labelnames = []
    labelIds = []
    for i in range(0, results.shape[0]):
        score = results[i, -2]
        if threshold and score < threshold:
            continue
        scores.append(score)
        label = int(results[i, -1])
        name = get_labelname(labelmap, label)[0]
        xmin = int(round(results[i, 0]))
        ymin = int(round(results[i, 1]))
        xmax = int(round(results[i, 2]))
        ymax = int(round(results[i, 3]))
        bboxes.append([xmin, ymin, xmax, ymax])
        labelIds.append(label)
        labelnames.append(name)
    return bboxes, scores, labelnames, labelIds


def np_to_caffe(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image / 255.0

class Refinedet512Model():
    def __init__(self,args):
        self.labelmap_file = args.labelmap_file
        self.model_def = args.model_def
        self.model_weights = args.model_weights
        self.category_index = ReadProTxt(self.labelmap_file)
        self.num_classes = len(self.category_index.items())
        self.net, self.transformer, self.labelmap = self.load_model()

    def load_model(self):
        caffe.set_device(0)
        caffe.set_mode_gpu()
        # load labelmap
        file = open(self.labelmap_file, 'r')
        labelmap = caffe_pb2.LabelMap()
        text_format.Merge(str(file.read()), labelmap)

        # load model

        net = caffe.Net(self.model_def, self.model_weights, caffe.TEST)

        # image preprocessing
        if '320' in self.model_def:
            img_resize = 320
        else:
            img_resize = 512
        net.blobs['data'].reshape(1, 3, img_resize, img_resize)
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
        transformer.set_raw_scale('data',
                                  255)  # the reference model operates on images in [0,255] range instead of [0,1]
        transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB

        # im_names = os.listdir('examples/images')

        return net, transformer, labelmap

    def predict(self,img,threshold=0.6):
        image = np_to_caffe(img)
        # image = caffe.io.load_image(image_file)
        transformed_image = self.transformer.preprocess('data', image)
        self.net.blobs['data'].data[...] = transformed_image

        detections = self.net.forward()['detection_out']

        det_label = detections[0, 0, :, 1]
        det_conf = detections[0, 0, :, 2]
        det_xmin = detections[0, 0, :, 3] * image.shape[1]
        det_ymin = detections[0, 0, :, 4] * image.shape[0]
        det_xmax = detections[0, 0, :, 5] * image.shape[1]
        det_ymax = detections[0, 0, :, 6] * image.shape[0]
        result = np.column_stack([det_xmin, det_ymin, det_xmax, det_ymax, det_conf, det_label])
        # show result
        bboxes, scores, labelnames, labels = ShowResults(image, result, self.labelmap, threshold)

        return bboxes,labelnames,labels,scores

    def predict_all_boxes(self, img, i, allboxes,threshold=0.6):
        image = np_to_caffe(img)
        # image = caffe.io.load_image(image_file)
        transformed_image = self.transformer.preprocess('data', image)
        self.net.blobs['data'].data[...] = transformed_image

        detections = self.net.forward()['detection_out']
        det_label = detections[0, 0, :, 1]
        det_conf = detections[0, 0, :, 2]
        det_xmin = detections[0, 0, :, 3] * image.shape[1]
        det_ymin = detections[0, 0, :, 4] * image.shape[0]
        det_xmax = detections[0, 0, :, 5] * image.shape[1]
        det_ymax = detections[0, 0, :, 6] * image.shape[0]
        result = np.column_stack([det_xmin, det_ymin, det_xmax, det_ymax, det_conf, det_label])

        boxes_out = []
        scores_out = []
        classes_out = []
        class_names_out = []

        for j in range(1, self.num_classes+1):
            dets = []
            for k in range(det_label.shape[0]):
                cId = int(det_label[k])
                class_name = self.category_index[cId]['name']
                score = det_conf[k]
                if cId == j and score > threshold:
                    ymin = int(det_ymin[k])
                    xmin = int(det_xmin[k])
                    ymax = int(det_ymax[k])
                    xmax = int(det_xmax[k])
                    boxes_out.append([xmin, ymin, xmax, ymax])
                    scores_out.append(score)
                    class_names_out.append(class_name)
                    classes_out.append(cId)
                    dets.append(np.reshape(np.array([xmin, ymin, xmax, ymax, score]), (1, 5)))
            if len(dets) == 0:
                allboxes[j][i] = np.empty([0, 5], dtype=np.float32)
            else:
                allboxes[j][i] = np.reshape(np.array(dets), (len(dets), 5))
        return boxes_out, class_names_out, classes_out, scores_out, allboxes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_def', type=str, default="/home/jade/Desktop/face-gesture-analysis-server/model/caffe_model_512/deploy.prototxt")
    parser.add_argument('--model_weights', type=str, default="/home/jade/Desktop/face-gesture-analysis-server/model/caffe_model_512/HAND_GESTURE_512_2018-12-29_refinedet_vgg16_512x512_iter_120000.caffemodel")
    parser.add_argument('--labelmap_file',type=str,default="/home/jade/Data/HandGesture/Hand_Gesture/hand_gesture.prototxt")
    args = parser.parse_args()
    refinedetModel = Refinedet512Model(args)
    image_paths = GetVOCTestImagePath("/home/jade/Data/HandGesture/Hand_Gesture")
    for image_path in image_paths:
        image = cv2.imread(image_path)
        bbooxes,label_text,classes,scores = refinedetModel.predict_all_boxes(image,)
        CVShowBoxes(image,bbooxes,label_text,classes,scores,waitkey=True)