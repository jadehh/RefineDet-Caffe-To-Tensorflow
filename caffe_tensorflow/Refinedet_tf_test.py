from jade import *
import argparse
from caffe_tensorflow.RefinedetModel  import Refinedet512Model

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--model_path', type=str, default="SDF/")
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--label_map_path', type=str, default="/home/jade/Data/StaticDeepFreeze/ThirtyTypes.pbtxt")
args = parser.parse_args()
refinedetModel = Refinedet512Model(args)

images = GetVOCTestImagePath("/home/jade/Data/StaticDeepFreeze/2019-03-18_14-11-36")
processbar = ProcessBar()
processbar.count = len(images)
start_time = time.time()
for image in images:
    processbar.start_time = time.time()
    name = GetLastDir(image)[:-4]
    image = cv2.imread(image)
    refinedetModel.print_var()
    bboxes_out, label_text, classes_out, scores_out, c_dets = refinedetModel.predict(image,0.6)
    #NoLinePrint("Detecting...",processbar)
    CVShowBoxes(image, bboxes_out,label_text,classes_out, scores_out, waitkey=True)
    #GenerateXml(name,image.shape,bboxes_out,classes_out,label_text,[0]*len(bboxes_out),[0]*len(bboxes_out),"/home/jade/Data/StaticDeepFreeze/2019-03-18_14-11-36/PredictAnnotations/"+name+".xml")
end_time = time.time()
print ("TIME:",(end_time-start_time)/float(len(images)))
