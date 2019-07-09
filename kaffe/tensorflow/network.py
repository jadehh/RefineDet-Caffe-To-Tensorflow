import os
import numpy as np
import tensorflow as tf
from math import *
from itertools import product as product
from utils.box_utils import decode, center_size
from layers.transformed_layer import l2_normalize_caffe
DEFAULT_PADDING = 'SAME'

def layer(op):
    '''Decorator for composable network layers.'''

    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated


class Network(object):

    def __init__(self, inputs, trainable=True):
        # The input nodes for this network
        self.inputs = inputs
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        self.trainable = trainable
        # Switch variable for dropout
        self.use_dropout = tf.placeholder_with_default(tf.constant(1.0),
                                                       shape=[],
                                                       name='use_dropout')
        self.setup()

    def setup(self):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')

    def save_ckpt(self,saver, sess, logdir,model_name,write_meta_graph=True):
        checkpoint_path = os.path.join(logdir, model_name)
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        saver.save(sess, checkpoint_path, write_meta_graph=write_meta_graph)
        print('The weights have been converted to {}.'.format(checkpoint_path))

    def load(self, data_path, session, ignore_missing=False):
        '''Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        '''
        data_dict = np.load(data_path,encoding="latin1").item()
        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                dict = data_dict[op_name]
                #conv4_3_norm/variance
                if type(dict) == list:
                    if op_name == "conv4_3_norm" or op_name == "conv5_3_norm":
                        var = tf.get_variable("weights")
                        session.run(var.assign(dict[0]))
                    if op_name == "P4-up" or op_name=="P5-up" or op_name=="P6-up":
                        dict1 = {}
                        dict1["weights"] = dict[0]
                        dict1["biases"] = dict[1]
                        for param_name, data in dict1.items():
                            var = tf.get_variable(param_name)
                            if param_name == "weights":
                                session.run(var.assign(np.transpose(data,[3,2,1,0])))
                            else:
                                session.run(var.assign(data))
                    else:
                        continue
                else:
                    for param_name, data in dict.items():
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))


    # def load(self, data_path, session, ignore_missing=False):
    #     '''Load network weights.
    #     data_path: The path to the numpy-serialized network weights
    #     session: The current TensorFlow session
    #     ignore_missing: If true, serialized weights for missing layers are ignored.
    #     '''
    #     data_dict = np.load(data_path).item()
    #     for op_name in data_dict:
    #         with tf.variable_scope(op_name, reuse=True):
    #             dict = data_dict[op_name]
    #             if type(dict) == list:
    #                 print op_name
    #             else:
    #                 for param_name, data in data_dict[op_name].iteritems():
    #                     try:
    #                         var = tf.get_variable(param_name)
    #                         session.run(var.assign(data))
    #                     except ValueError:
    #                         if not ignore_missing:
    #                             raise

    def feed(self, *args):
        '''Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        '''
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, str):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self):
        '''Returns the current network output.'''
        #, ,
        #            'arm_priorbox',
        #            'arm_conf_flatten',
        #            'arm_loc')
        return self.layers["detection_out"]
    def get_unique_name(self, prefix):
        '''Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        '''
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def make_var(self, name, shape):
        '''Creates a new TensorFlow variable.'''
        return tf.get_variable(name, shape, trainable=self.trainable)

    def validate_padding(self, padding):
        '''Verifies that the padding is one of the supported ones.'''
        assert padding in ('SAME', 'VALID')

    @layer
    def conv(self,
             input,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             padding=DEFAULT_PADDING,
             pad=None,
             group=1,
             dilation=None,
             biased=True):
        input = tf.cast(input, tf.float32)
        if pad == 1:
            padding = "VALID"
            shape = input.get_shape()
            input = tf.pad(input, ((0, 0), (pad, 0), (pad, 0), (0, 0)), 'constant', constant_values=0)
            input = tf.reshape(input, [-1, shape[1] + pad, shape[2] + pad, shape[3]])
        if pad == 3:
            padding = "VALID"
            shape = input.get_shape()
            input = tf.pad(input, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)
            input = tf.reshape(input, [-1, shape[1] + pad * 2, shape[2] + pad * 2, shape[3]])
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = int(input.get_shape()[-1])
        # Verify that the grouping parameter is valid
        assert c_i % group == 0
        assert c_o % group == 0
        # Convolution for a given input and kernel
        if dilation:
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], dilations=[1, dilation, dilation, 1],
                                                 padding=padding)
        else:
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i / group, c_o])
            if group == 1:
                # convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
                # This is the common-case. Convolve the input without any further complications.
                output = convolve(input, kernel)
            else:
                # Split the input into groups and then convolve each of them independently
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                # Concatenate the groups
                output = tf.concat(3, output_groups)
            # Add the biases
            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)
            return output


    #2, 2, 256, 2, 2, padding = None, relu = False, name = 'P6-up')
    @layer
    def dconv(self,
              input,
              k_h,
              k_w,
              c_o,
              s_h,
              s_w,
              name,
              relu=True,
              padding=DEFAULT_PADDING,
              group=1,
              biased=True):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = int(input.get_shape()[-1])
        # Verify that the grouping parameter is valid
        assert c_i % group == 0
        assert c_o % group == 0

        o_s = int(input.shape[1])
        output_shape = [1, k_w * o_s, k_w * o_s, 256]

        # conv2d_transpose(value, filter, output_shape, strides, padding="SAME",
        # data_format="NHWC", name=None)
        deconvolve = lambda i, k: tf.nn.conv2d_transpose(i, k, output_shape, [1, s_h, s_w, 1], padding="SAME",
                                                         data_format="NHWC", name=None)
        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i / group, c_o])
            if group == 1:
                # This is the common-case. Convolve the input without any further complications.
                output = deconvolve(input, kernel)
            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)
            return output


    @layer
    def permute(self,input,t1,t2,t3,t4,name):
        with tf.variable_scope(name) as scope:
            output = tf.transpose(input,(t1,t2,t3,t4))
        return output

    @layer
    def flatten(self,input,flat,name):
        with tf.variable_scope(name) as scope:
            output =tf.layers.flatten(input,name=name)
        return output

    @layer
    def prior_box(self,input,anchor_aspect_size,anchor_ratios,anchor_steps,anchor_offset,name):
        mean = []
        image = input[1]
        layer = input[0]
        image_size = image.shape.as_list()[2]
        width = []
        features_maps = [layer.shape.as_list()[2]]
        for k, f in enumerate(features_maps):
            for i, j in product(range(f), repeat=2):
                f_k = image_size / float(anchor_steps)
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = anchor_aspect_size[0] / image_size

                mean += [cx, cy, s_k, s_k]
                width += [0.1, 0.1, 0.2, 0.2]
                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                if len(anchor_aspect_size)==2:
                    s_k_prime = sqrt(s_k * (anchor_aspect_size[1] / image_size))
                    mean += [cx - s_k_prime * 0.5, cy - s_k_prime * 0.5, cx + s_k_prime * 0.5, cy + s_k_prime * 0.5]
                    width += [0.1, 0.1, 0.2, 0.2]
                # rest of aspect ratios
                for ar in anchor_ratios:
                    width += [0.1, 0.1, 0.2, 0.2]

                    mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    width += [0.1, 0.1, 0.2, 0.2]
                    mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]
        # back to torch land

        output = np.array(mean)
        output = tf.convert_to_tensor(np.resize(output, [1, len(mean)]), dtype="float32", name=name)
        return output

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING,pad=0):
        if pad == 1 :
            shape = input.get_shape()
            input = tf.pad(input, ((0, 0),(pad, 0), (pad, 0), (0, 0)), 'constant', constant_values=0)
            input = tf.reshape(input,[-1,shape[1]+pad,shape[2]+pad,shape[3]])
            padding="VALID"
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(values=inputs,axis=axis,name=name)
    @layer
    def reshape(self,inputs,t1,t2,t3,name):
        return tf.reshape(inputs, [t1, t2 , t3],name=name)
    @layer
    def add(self, inputs,name):
        return tf.add_n(inputs, name=name)

    @layer
    def fc(self, input, num_out, name, relu=True):
        with tf.variable_scope(name) as scope:
            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                # The input is spatial. Vectorize it first.
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(input, [-1, dim])
            else:
                feed_in, dim = (input, input_shape[-1].value)
            weights = self.make_var('weights', shape=[dim, num_out])
            biases = self.make_var('biases', [num_out])
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def softmax(self, input, name):
        return tf.nn.softmax(input, name=name)

    @layer
    def batch_normalization_caffe(self, input, name, scale_offset=True, relu=False,epsilon=1e-10,dim=1):
        # NOTE: Currently, only inference is supported
        with tf.variable_scope(name) as scope:
            shape = [input.get_shape()[-1]]
            if scale_offset:
                scale = self.make_var('scale', shape=shape)
                offset = self.make_var('offset', shape=shape)
            else:
                scale, offset = (None, None)
            output = l2_normalize_caffe(x=input,
                                              epsilon=epsilon,
                                              pow=2,
                                              dim=3,
                                              keepdims=True,
                                              name=name)
            weights = self.make_var('weights', shape=shape)
            input /= output
            weights1 = tf.expand_dims(tf.expand_dims(tf.expand_dims(weights, 0), 0), 0)
            x1_expand = tf.tile(weights1, [1, input.get_shape()[2], input.get_shape()[2], 1])
            out1 = x1_expand * input
            if relu:
                out1 = tf.nn.relu(out1)
            return out1


    @layer
    def batch_normalization_caffe(self, input, name, scale_offset=True, relu=False,epsilon=1e-10,dim=1):
        # NOTE: Currently, only inference is supported
        with tf.variable_scope(name) as scope:
            shape = [input.get_shape()[-1]]
            if scale_offset:
                scale = self.make_var('scale', shape=shape)
                offset = self.make_var('offset', shape=shape)
            else:
                scale, offset = (None, None)
            output = l2_normalize_caffe(x=input,
                                              epsilon=epsilon,
                                              pow=2,
                                              dim=3,
                                              keepdims=True,
                                              name=name)
            weights = self.make_var('weights', shape=shape)
            input /= output
            weights1 = tf.expand_dims(tf.expand_dims(tf.expand_dims(weights, 0), 0), 0)
            x1_expand = tf.tile(weights1, [1, input.get_shape()[2], input.get_shape()[2], 1])
            out1 = x1_expand * input
            if relu:
                out1 = tf.nn.relu(out1)
            return out1


    @layer
    def batch_normalization(self, input, name, scale_offset=True, relu=False, epsilon=1e-10, dim=1):
        # NOTE: Currently, only inference is supported
        with tf.variable_scope(name) as scope:
            shape = [input.get_shape()[-1]]
            if scale_offset:
                scale = self.make_var('weights', shape=shape)
                offset = self.make_var('biases', shape=shape)
            else:
                scale, offset = (None, None)
            output = tf.nn.batch_normalization(
                input,
                mean=self.make_var('running_mean', shape=shape),
                variance=self.make_var('running_var', shape=shape),
                offset=offset,
                scale=scale,
                # TODO: This is the default Caffe batch norm eps
                # Get the actual eps from parameters
                variance_epsilon=1e-5,
                name=name)
            if relu:
                output = tf.nn.relu(output)
            return output



    @layer
    def dropout(self, input, keep_prob, name):
        keep = 1 - self.use_dropout + (self.use_dropout * keep_prob)
        return tf.nn.dropout(input, keep, name=name)

    # #3, True, 0, 0.449999988079, 1000, 2, 500, 0.00999999977648, 0.00999999977648,
    @layer
    def detection_out(self,input,num_classes,objectness_score,name):
        variance = [0.1,0.2]
        loc, conf = input[0],input[1]
        prior_data = input[2]
        arm_loc, arm_conf = input[4], input[3]
        arm_loc = tf.reshape(arm_loc, [arm_loc.shape[0], -1, 4])
        arm_conf = tf.reshape(arm_conf, [-1, 2])
        # conf preds
        loc = tf.reshape(loc, [loc.shape[0], -1, 4])
        conf = tf.reshape(conf, [-1, num_classes])

        prior_data = tf.reshape(prior_data, [-1, 4])
        loc_data = loc
        conf_data = conf

        num = loc_data.shape[0]  # batch size

        arm_loc_data = arm_loc
        arm_conf_data = arm_conf
        arm_object_conf = arm_conf_data[:, 1:]
        no_object_index = arm_object_conf <= objectness_score
        expands = tf.tile(no_object_index, [1, num_classes])
        conf_data = tf.where(expands, tf.zeros_like(conf_data), conf_data)


        num_priors = prior_data.shape[0]

        if num == 1:
            # size batch x num_classes x num_priors
            conf_preds = tf.expand_dims(conf_data, 0)

        # Decode predictions into bboxes.
        for i in range(num):
            default = decode(arm_loc_data[i], prior_data, variance)
            default = center_size(default)
            decoded_boxes = decode(loc_data[i], default, variance)
            # For each class, perform nms
            conf_scores = conf_preds[i]

            boxes = tf.expand_dims(decoded_boxes, 0)
            scores = tf.expand_dims(conf_scores, 0)
        return boxes, scores

    @layer
    def cut_img(self,input,name):
        img = input[1]
        box = input[0][0]
        scores = input[0][1]
        img = tf.cast(img, dtype=tf.float32)
        scale = ([int(img.get_shape()[1]), int(img.get_shape()[0]),
                  int(img.get_shape()[1]), int(img.get_shape()[1])])
        boxes = box[0]
        scores = scores[0]
        boxes *= scale
        # scale each detection back up to the image
        for j in range(1, 1 + 1):
            inds = tf.where(scores[:, j] > 0.9)

            c_bboxes = tf.gather_nd(boxes, inds)
            c_scores = tf.gather_nd(scores[:, j], inds)
            c_dets = tf.concat((c_bboxes, c_scores[:, np.newaxis]), axis=1)
            keep = tf.image.non_max_suppression(c_bboxes, c_scores, max_output_size=10, iou_threshold=0.45)
            keep = tf.reshape(keep, [-1, 1])
            c_dets = tf.gather_nd(c_dets, keep)
            c_bboxes = tf.gather_nd(c_bboxes, keep)
            c_bboxes = tf.cast(c_bboxes, tf.int32)
            c_bboxes = tf.where(c_bboxes < 0, tf.zeros_like(c_bboxes), y=c_bboxes)
            cnn = tf.map_fn(fn=lambda inp: tf.image.resize_images(img[inp[1]:inp[3], inp[0]:inp[2], :],[128,384]),
                            elems=c_bboxes,
                            dtype=tf.float32)
            return tf.cast(cnn,dtype=tf.uint8)

    @layer
    def cut_img(self,input,name):
        img = input[1]
        img = tf.reshape(img,[1,img.get_shape()[0],img.get_shape()[1],img.get_shape()[2]])
        img = tf.image.resize_images(img,[128,384])
        return tf.cast(img,dtype=tf.uint8)

    @layer
    def p_transpose(self,input,k,name):
        with tf.variable_scope(name) as scope:
            z0_p2 = input[:, :, k:k+1, :]
        return z0_p2

    @layer
    def squeeze(self,input,dims,name):
        for dim in dims:
            input = tf.squeeze(input,dim,name)
        return input

