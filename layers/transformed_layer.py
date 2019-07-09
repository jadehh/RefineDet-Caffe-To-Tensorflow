import numpy as np
import cv2
import tensorflow as tf
import skimage.io as sio
from scipy.ndimage import zoom
from skimage.transform import resize
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops


def l2_normalize_caffe(x,epsilon=1e-10,pow=2,dim=None,keepdims=None,name=None):
    with ops.name_scope(name, "l2_normalize_caffe", [x]) as name:
      x = ops.convert_to_tensor(x, name="x")
      x_pow = math_ops.pow(x, pow)
      x_sum = math_ops.reduce_sum(x_pow,[dim],keepdims=True)
      x_sqrt = math_ops.sqrt(x_sum)
      x = math_ops.add(x_sqrt,epsilon)
      return x


def resize_image(im, new_dims, interp_order=1):
    if im.shape[-1] == 1 or im.shape[-1] == 3:
        im_min, im_max = im.min(), im.max()
        if im_max > im_min:
            # skimage is fast but only understands {1,3} channel images
            # in [0, 1].
            im_std = (im - im_min) / (im_max - im_min)
            resized_std = resize(im_std, new_dims, order=interp_order)
            resized_im = resized_std * (im_max - im_min) + im_min
        else:
            # the image is a constant -- avoid divide by 0
            ret = np.empty((new_dims[0], new_dims[1], im.shape[-1]),
                           dtype=np.float32)
            ret.fill(im_min)
            return ret
    else:
        # ndimage interpolates anything but more slowly.
        scale = tuple(np.array(new_dims, dtype=float) / np.array(im.shape[:2]))
        resized_im = zoom(im, scale + (1,), order=interp_order)
    return resized_im.astype(np.float32)

def transformed_image(image,default_size=300):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    transpose = [2,0,1]
    channel_swap = [2, 1, 0]
    raw_scale = 255
    mean = np.array([104, 117, 123])
    mean = np.resize(mean,[3,1,1])

    if image.shape[2] != default_size:
        im = resize(image,(default_size,default_size))

    if transpose is not None:
        im = im.transpose(transpose)
    if channel_swap is not None:
        im = im[channel_swap, :, :]
    if raw_scale is not None:
        im *= raw_scale
    if mean is not None:
        im -= mean
    return im


def transformed_image_np(image,default_size=512):
    mean = np.array([123, 117, 104])
    mean = np.resize(mean,[3,1,1])
    transpose = [2, 0, 1]
    if image.shape[2] != default_size:
        im = cv2.resize(image,(default_size,default_size)).astype(np.float32)
    if transpose is not None:
        im = im.transpose(transpose)
    if mean is not None:
        im -= mean
    im = np.array([im])
    im = np.transpose(im,[0,3,2,1])
    return im

def transformed_image_tf(image,default_size=512):
    mean = np.array([123, 117, 104])
    mean = np.resize(mean,[3,1,1])
    im = tf.image.resize_images(image, (default_size, default_size))
    transpose = [2, 0, 1]
    if transpose is not None:
        im = tf.transpose(im,transpose)
    im = im - mean
    im = tf.reshape(im,[1,im.shape[0],im.shape[1],im.shape[2]])
    im = tf.transpose(im, [0,3,2,1])
    return im


if __name__ == '__main__':
    image = sio.imread("/home/jade/Desktop/SSD_Tensorflow_Caffe/images/fish-bike.jpg")
    image = transformed_image(image)

    image2 = cv2.imread("/home/jade/Desktop/SSD_Tensorflow_Caffe/images/fish-bike.jpg")
    # image2 = image2 - np.array([1,0,-1])
    # image2 = image2 / 255.0
    image2_resize_np = transformed_image_np(image2,default_size=512)
    image2_resize_tf = transformed_image_tf(tf.convert_to_tensor(image2),default_size=512)
    with tf.Session() as sess:
        out = sess.run(image2_resize_tf)
   # tmp = image2_resize_np - image2_resize
# transformed_image = np.load("transformed_image.npy")
# transformed_image = np.array([transformed_image])
# transformed_image = np.transpose(transformed_image,[0,3,2,1])