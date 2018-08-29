from __future__ import absolute_import
from __future__ import print_function

import os
import six
from test_conversion_imagenet import TestModels

def get_test_table():
    if six.PY2:
        return {
            'keras' : {
                'vgg19'        : [CaffeEmit, CntkEmit, CoreMLEmit, KerasEmit, MXNetEmit, PytorchEmit, TensorflowEmit],
                'inception_v3' : [CaffeEmit, CntkEmit, CoreMLEmit, KerasEmit, MXNetEmit, PytorchEmit, TensorflowEmit],
                'resnet50'     : [CaffeEmit, CntkEmit, CoreMLEmit, KerasEmit, MXNetEmit, PytorchEmit, TensorflowEmit],
                'densenet'     : [CaffeEmit, CntkEmit, CoreMLEmit, KerasEmit, MXNetEmit, PytorchEmit, TensorflowEmit],
                'xception'     : [TensorflowEmit, KerasEmit, CoreMLEmit],
                'mobilenet'    : [CoreMLEmit, KerasEmit, TensorflowEmit], # TODO: MXNetEmit
                # 'nasnet'       : [TensorflowEmit, KerasEmit, CoreMLEmit],
            }},
    else:
        return None


def test_keras():
    tester = TestModels()
    tester._test_function('keras', tester.KerasParse)