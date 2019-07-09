from jade.voc0712 import VOCDetection,AnnotationTransform

testset = VOCDetection(
    "/home/jade/Data/HandGesture/", [('Hand_Gesture', 'test_var')], None, AnnotationTransform())
from jade import *
import argparse
from RefinedetModel import Refinedet512Model

parser = argparse.ArgumentParser()
parser.add_argument('--model_def', type=str,
                    default="/home/jade/Desktop/face-gesture-analysis-server/model/caffe_model_512/deploy.prototxt")
parser.add_argument('--model_weights', type=str,
                    default="/home/jade/Desktop/face-gesture-analysis-server/model/caffe_model_512/HAND_GESTURE_512_2018-12-29_refinedet_vgg16_512x512_iter_120000.caffemodel")
parser.add_argument('--labelmap_file', type=str,
                    default="/home/jade/Data/HandGesture/Hand_Gesture/hand_gesture.prototxt")
args = parser.parse_args()

refinedetModel = Refinedet512Model(args)

num_images = len(testset)
num_classes = 2 + 1
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
    bboxes_out, label_text, classes_out, scores_out, allboxes = refinedetModel.predict_all_boxes(img,i,all_boxes,0.6)
    #CVShowBoxes(img,bboxes_out,label_text,classes_out,scores_out,waitkey=True)
    NoLinePrint("Detecting ...",processbar)
    if max_per_image > 0:
        image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1, num_classes)])
        if len(image_scores) > max_per_image:
            image_thresh = np.sort(image_scores)[-max_per_image]
            for j in range(1, num_classes):
                keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                all_boxes[j][i] = all_boxes[j][i][keep, :]


testset.evaluate_detections(all_boxes)