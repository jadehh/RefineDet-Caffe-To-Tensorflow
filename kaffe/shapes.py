import math
from collections import namedtuple

from .errors import KaffeError

TensorShape = namedtuple('TensorShape', ['batch_size', 'channels', 'height', 'width'])
TensorShape_2 = namedtuple("TensorShape_2",['batch_size','count'])
TensorShape_3 = namedtuple("TensorShape_3",['batch_size','center','count'])
def get_filter_output_shape(i_h, i_w, params, round_func):
    o_h = (i_h ) / float(params.stride_h)
    o_w = (i_w ) / float(params.stride_w)
    return (int(round_func(o_h)), int(round_func(o_w)))

def get_permute_output_shape(node):
    params = node.layer.parameters
    orders = params.order
    output_shape = node.parents[0].output_shape
    shape =  TensorShape(output_shape[int(orders[0])],output_shape[int(orders[1])],output_shape[int(orders[2])],output_shape[int(orders[3])])
    return shape


def get_priorbox_shape(node):
    params = node.layer.parameters
    output_shape = node.parents[0].output_shape
    return TensorShape_3(output_shape[0],2,output_shape[2]*output_shape[3]*12)
def get_reshape_shape(node):
    params = node.layer.parameters
    dim = params.shape.dim
    output_shape = node.parents[0].output_shape
    return TensorShape_3(output_shape[0],output_shape[1]/int(dim[2]), int(dim[2]))
def get_flatten_shape(node):
    params = node.layer.parameters
    output_shape = node.parents[0].output_shape
    count = 1
    for i in range(len(output_shape)):
        count = count * output_shape[i]
    return TensorShape_2(output_shape[0],count)
def get_deconvolution_shape(node):
    output_shape = node.parents[0].output_shape
    return TensorShape(int(output_shape[0]),int(output_shape[1]),int(output_shape[2]*2),int(output_shape[3]*2))
def get_strided_kernel_output_shape(node, round_func):
    assert node.layer is not None
    input_shape = node.get_only_parent().output_shape
    o_h, o_w = get_filter_output_shape(input_shape.height, input_shape.width,
                                       node.layer.kernel_parameters, round_func)
    params = node.layer.parameters
    has_c_o = hasattr(params, 'num_output')
    c = params.num_output if has_c_o else input_shape.channels

    return TensorShape(input_shape.batch_size, c, o_h, o_w)

def write_txt(node,shape):
    with open("shape_false.txt", 'a') as f:
        if len(shape) == 4:
            f.write("name:"+node.name+" " +" output_shape:"+ "["+str(shape[0])+","+str(shape[1])+","+str(shape[2])+","+str(shape[3])+"]" +"\n")
        elif len(shape) == 3:
            f.write(
                "name:" + node.name + " " + " output_shape:" + "[" + str(shape[0]) + "," + str(shape[1])+","+str(shape[2]) + "]" + "\n")
        elif len(shape) == 2:
            f.write(
                "name:" + node.name + " " + " output_shape:" + "[" + str(shape[0]) + "," + str(shape[1]) + "]" + "\n")


def shape_not_implemented(node):
    raise NotImplementedError


def shape_identity(node):
    assert len(node.parents) > 0
    shape = node.parents[0].output_shape
    write_txt(node,shape)
    return shape

def shape_normalize(node):
    assert len(node.parents) > 0
    shape = node.parents[0].output_shape
    write_txt(node,shape)
    return shape

def shape_flatten(node):
    assert len(node.parents) > 0
    shape = get_flatten_shape(node)
    write_txt(node,shape)
    return shape
def shape_permute(node):
    assert len(node.parents) > 0
    shape = get_permute_output_shape(node)
    write_txt(node,shape)
    return shape
def shape_priorbox(node):
    assert len(node.parents) > 0
    shape = get_priorbox_shape(node)
    write_txt(node,shape)
    return shape
def shape_reshape(node):
    assert len(node.parents) > 0
    shape = get_reshape_shape(node)
    write_txt(node,shape)
    return shape
def shape_detection_out(node):
    assert len(node.parents) > 0
    shape = node.parents[0].output_shape
    write_txt(node,shape)
    return shape

def shape_scalar(node):
    shape = TensorShape(1, 1, 1, 1)
    write_txt(node,shape)
    return shape



def shape_data(node):
    if node.output_shape:
        # Old-style input specification
        shape = node.output_shape
        write_txt(node,shape)
        return shape
    try:
        # New-style input specification
        return map(int, node.parameters.shape[0].dim)
    except:
        # We most likely have a data layer on our hands. The problem is,
        # Caffe infers the dimensions of the data from the source (eg: LMDB).
        # We want to avoid reading datasets here. Fail for now.
        # This can be temporarily fixed by transforming the data layer to
        # Caffe's "input" layer (as is usually used in the "deploy" version).
        # TODO: Find a better solution for this.
        raise KaffeError('Cannot determine dimensions of data layer.\n'
                         'See comments in function shape_data for more info.')


def shape_mem_data(node):
    params = node.parameters
    shape = TensorShape(params.batch_size, params.channels, params.height, params.width)
    write_txt(node,shape)
    return shape


def shape_concat(node):
    axis = node.layer.parameters.axis
    output_shape = None
    for parent in node.parents:
        if output_shape is None:
            output_shape = list(parent.output_shape)
        else:
            output_shape[axis] += parent.output_shape[axis]
    shape = tuple(output_shape)
    write_txt(node,shape)
    return shape




def shape_convolution(node):
    # intput:(1, 3, 320, 320)
    # params:(64, 3, 3, 3)
    shape = get_strided_kernel_output_shape(node, math.floor)
    write_txt(node,shape)
    return shape

def shape_deconvolution(node):
    # intput:(1, 256, 14, 14)
    # parmas:(256, 256, 2, 2)
    # output:(1, 256, 7, 7)
    shape = get_deconvolution_shape(node)
    write_txt(node,shape=shape)
    return shape

def shape_pool(node):
    shape = get_strided_kernel_output_shape(node, math.ceil)
    write_txt(node,shape=shape)
    return shape


def shape_inner_product(node):
    input_shape = node.get_only_parent().output_shape
    shape = TensorShape(input_shape.batch_size, node.layer.parameters.num_output, 1, 1)
    write_txt(node,shape)
    return shape
