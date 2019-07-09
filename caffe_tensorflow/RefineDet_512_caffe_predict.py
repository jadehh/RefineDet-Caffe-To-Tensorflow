from jade import *
import argparse
from caffe_tensorflow.RefinedetModelNpy  import Refinedet512Model
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--model_path', type=str, default="StaticDeepFreeze/StaticDeepFreeze.npy")
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--label_map_path', type=str, default="/home/jade/label_map/wild_goods.prototxt")
args = parser.parse_args()
refinedetModel = Refinedet512Model(args)

images = GetAllImagesPath("/home/jade/Data/StaticDeepFreeze/2019-03-18_14-11-36/JPEGImages")
processbar = ProcessBar()
processbar.count = len(images)
start_time = time.time()
for image in images:
    processbar.start_time = time.time()
    image = cv2.imread(image)
    bboxes_out, label_text, classes_out, scores_out, c_dets = refinedetModel.predict(image)
    NoLinePrint("Detecting...",processbar)
    CVShowBoxes(image, bboxes_out,label_text,classes_out, scores_out, waitkey=True)
end_time = time.time()
print ("TIME:",(end_time-start_time)/float(len(images)))
