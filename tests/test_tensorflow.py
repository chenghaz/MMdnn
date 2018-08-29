from __future__ import absolute_import
from __future__ import print_function

import os
from test_conversion_imagenet import TestModels

def get_test_table():
    TRAVIS_CI = os.environ.get('TRAVIS')
    if not TRAVIS_CI or TRAVIS_CI.lower() != 'true':
        return None

    ONNX = os.environ.get('TEST_ONNX')
    if ONNX and ONNX.lower() == 'true':
        return None

    return { 'tensorflow' :
    {
        'vgg19'                : [CaffeEmit, CoreMLEmit, CntkEmit, KerasEmit, MXNetEmit, PytorchEmit],
        'inception_v1'         : [CaffeEmit, CoreMLEmit, KerasEmit, MXNetEmit, PytorchEmit, TensorflowEmit],
        'inception_v3'         : [CaffeEmit, CoreMLEmit, CntkEmit, KerasEmit, MXNetEmit, PytorchEmit, TensorflowEmit],
        'resnet_v1_152'        : [CaffeEmit, CoreMLEmit, KerasEmit, MXNetEmit, PytorchEmit, TensorflowEmit],
        'resnet_v2_152'        : [CaffeEmit, CoreMLEmit, CntkEmit, KerasEmit, MXNetEmit, PytorchEmit, TensorflowEmit],
        'mobilenet_v1_1.0'     : [CoreMLEmit, CntkEmit, KerasEmit, MXNetEmit, PytorchEmit, TensorflowEmit],
        'mobilenet_v2_1.0_224' : [CoreMLEmit, CntkEmit, KerasEmit, MXNetEmit, PytorchEmit, TensorflowEmit],
        # 'nasnet-a_large'       : [MXNetEmit, PytorchEmit, TensorflowEmit],
        # 'inception_resnet_v2'  : [CaffeEmit, KerasEmit, MXNetEmit, PytorchEmit, TensorflowEmit],
    }}


def test_tensorflow():
    test_table = get_test_table()
    tester = TestModels(test_table)
    tester._test_function('tensorflow', tester.TensorFlowParse)