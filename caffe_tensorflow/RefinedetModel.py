#coding=utf-8
# Import the converted model's class
from caffe_tensorflow.StaticDeepFreeze.StaticDeepFreeze import StaticDeepFreeze
import tensorflow as tf
from utils.nms_wrapper import nms
from jade import *
from layers.transformed_layer import transformed_image_tf
import argparse
class Refinedet512Model():
    def __init__(self,args):
        self.model_path = args.model_path
        self.num_classes = args.num_classes
        self.categories,self.label_map = ReadProTxt(args.label_map_path)
        self.net, self.sess = self.load_model()



    def print_var(self):
        variable_names = [v.name for v in tf.trainable_variables()]
        values = self.sess.run(variable_names)
        for k, v in zip(variable_names, values):
            print("Variable: ", k)
            print("Shape: ", v.shape)
            print(v)


    def load_model(self):
        sess = tf.Session()
        input = tf.placeholder(tf.float32, (None, None, 3), 'input')
        input = transformed_image_tf(input, default_size=512)
        net = StaticDeepFreeze({'data': input})
        # net.load(self.model_path,sess)
        saver = tf.train.Saver()
        saver.restore(sess,tf.train.latest_checkpoint(self.model_path))
        return net,sess


    def predict(self,img,threshold=0.6):
        if type(img) == str:
            img = cv2.imread(img)
        boxes, scores = self.sess.run(self.net.get_output(),feed_dict={'input:0': img})
        scale = ([img.shape[1], img.shape[0],
                  img.shape[1], img.shape[0]])
        boxes = boxes[0]
        scores = scores[0]
        boxes *= scale
        label_text = []
        labels = []
        bboxes_out = []
        scores_out = []
        classes_out = []
        # scale each detection back up to the image
        for j in range(1, self.num_classes+1):
            inds = np.where(scores[:, j] > 0.45)[0]
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)
            keep = nms(c_dets, 0.45, force_cpu=True)
            c_dets = c_dets[keep, :]
            for i in range(len(c_dets)):
                box = [c_dets[i][0], c_dets[i][1], c_dets[i][2], c_dets[i][3]]
                bboxes_out.append(box)
                scores_out.append(c_dets[i][4])
                classes_out.append(j)
        for cls_id in classes_out:
            if cls_id in self.categories:
                class_name = self.categories[cls_id]['name']
                label_text.append(class_name)
        return bboxes_out,label_text,classes_out,scores_out,c_dets
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--model_path', type=str, default="models/Refinedet.npy")
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--label_map_path',type=str,default="/home/jade/label_map/hand_gesture_label_map.pbtxt")
    args = parser.parse_args()
    refinedetModel = Refinedet512Model(args)
    image = cv2.imread(
        "/home/jade/Data/HandGesture/done/v1_2018-12-20_14-22-21/JPEGImages/0bfaee80-0450-11e9-a71b-88d7f6413e60.jpg")
    bbooxes,label_text,classes,scores = refinedetModel.predict(image)
    CVShowBoxes(image,bbooxes,label_text,classes,scores,waitkey=True)