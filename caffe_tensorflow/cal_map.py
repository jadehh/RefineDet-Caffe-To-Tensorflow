from jade.voc0712 import VOCDetection,AnnotationTransform
import numpy as np
testset = VOCDetection(
    "/home/jade/Data/HandGesture", [('Hand_Gesture', 'test')], None, AnnotationTransform())
from jade.jade_tools import *
import argparse
from RefinedetModel import Refinedet512Model
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--model_path', type=str, default="tensorflow_model")
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--label_map_path', type=str, default="/home/jade/label_map/hand_gesture_label_map.pbtxt")
args = parser.parse_args()
refinedetModel = Refinedet512Model(args)

num_images = len(testset)
num_classes = args.num_classes + 1
all_boxes = [[[] for _ in range(num_images)]
             for _ in range(num_classes)]

max_per_image = 300
processbar = ProcessBar()
processbar.count = num_images
for i in range(num_images):
    processbar.start_time = time.time()
    img = testset.pull_image(i)
    scale = ([img.shape[1], img.shape[0],
                          img.shape[1], img.shape[0]])
    bboxes_out, label_text, classes_out, scores , c_dets= refinedetModel.predict(img)
    NoLinePrint("Detecting ...",processbar)
    for j in range(1, num_classes):
        all_boxes[j][i] = c_dets
    if max_per_image > 0:
        image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1, num_classes)])
        if len(image_scores) > max_per_image:
            image_thresh = np.sort(image_scores)[-max_per_image]
            for j in range(1, num_classes):
                keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                all_boxes[j][i] = all_boxes[j][i][keep, :]

testset.evaluate_detections(all_boxes)