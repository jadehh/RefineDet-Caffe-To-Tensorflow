#coding=utf-8
import tensorflow as tf
import math
import numpy as np




def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return tf.concat(((boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2]), -1)  # w, h



# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = tf.concat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * tf.exp(loc[:, 2:] * variances[1])), 1)
    tmp1 = boxes[:, :2] - (boxes[:, 2:] / 2)
    tmp2 = boxes[:, 2:] + tmp1
    tmp3 = tf.concat((tmp1,tmp2),-1)
    return tmp3
    #boxes = np.concatenate((
    #    priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
    #    priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
    #boxes[:, :2] -= boxes[:, 2:] / 2
    #boxes[:, 2:] += boxes[:, :2]
    #return boxes




