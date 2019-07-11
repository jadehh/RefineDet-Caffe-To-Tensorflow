#!/usr/bin/env python

import os
import sys
import numpy as np
import argparse
from kaffe import KaffeError, print_stderr
from kaffe.tensorflow import TensorFlowTransformer
from jade import *

def fatal_error(msg):
    print_stderr(msg)
    exit(-1)


def validate_arguments(args):
    if (args.data_output_path is not None) and (args.caffemodel is None):
        fatal_error('No input data path provided.')
    if (args.caffemodel is not None) and (args.data_output_path is None):
        fatal_error('No output data path provided.')
    if (args.code_output_path is None) and (args.data_output_path is None):
        fatal_error('No output path specified.')


def convert(class_name,def_path, caffemodel_path, data_output_path, code_output_path, phase,num_classes):
    transformer = TensorFlowTransformer(class_name,def_path, caffemodel_path, num_classes,phase=phase)
    print_stderr('Converting data...')
    if caffemodel_path is not None:
        data = transformer.transform_data()
        print_stderr('Saving data...')
        with open(data_output_path, 'wb') as data_out:
            np.save(data_out, data)
    if code_output_path:
        print_stderr('Saving source...')
        with open(code_output_path, 'wb') as src_out:
            src_out.write(transformer.transform_source())
    print_stderr('Done.')



def main():
    deploy_path = "/data/home/jdh/models/VOC0712Plus/refinedet_vgg16_512x512_ft/deploy.prototxt"
    caffemodel_path = "/data/home/jdh/models/VOC0712Plus/refinedet_vgg16_512x512_ft/VOC0712Plus_refinedet_vgg16_512x512_ft_final.caffemodel"
    dataname = "VOC0712Plus"
    num_classes = 21

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', default=num_classes, help='num_classes')
    parser.add_argument('--def_path', default=deploy_path,help='Model definition (.prototxt) path')
    parser.add_argument('--datasetname', default=dataname,help='Model definition (.prototxt) path')
    parser.add_argument('--caffemodel',default=caffemodel_path, help='Model data (.caffemodel) path')
    CreateSavePath("caffe_tensorflow/")
    CreateSavePath("caffe_tensorflow/"+dataname)
    parser.add_argument('--data-output-path',default="caffe_tensorflow/"+dataname+"/"+dataname+".npy",help='Converted data output path')
    parser.add_argument('--code-output-path',default="caffe_tensorflow/"+dataname+"/"+dataname+".py",help='Save generated source to this path')
    parser.add_argument('-p',
                        '--phase',
                        default='test',
                        help='The phase to convert: test (default) or train')
    args = parser.parse_args()
    validate_arguments(args)
    convert(dataname,args.def_path, args.caffemodel, args.data_output_path, args.code_output_path,
            args.phase,args.num_classes)
    with open("caffe_tensorflow/"+dataname+"/"+"__init__.py",'w') as f:
        f.write("\n")


if __name__ == '__main__':
    main()
