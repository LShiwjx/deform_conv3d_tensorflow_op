from __future__ import absolute_import
import tensorflow as tf
import os.path as osp
from tensorflow.python.framework import ops

# python wrapper
filename = osp.join(osp.dirname(__file__), 'deform_conv3d.so')
_deform_conv3d_module = tf.load_op_library(filename)
"""
Args:
    Forward:NCLHW
    Filter:N'CL'H'W'
    Offset:GL"H"W"L'H'W'3
Attrs:
    strides: S
    dilatation_rates: D
    padding: VALID or SAME
        VALID: L"=(L-L'+1)/S P=0
        SAME: L"=ceil(L/S) P=((L"-1)*S+L'-L)/2
Return:
    Backward:NC*C'L"H"W"
"""
deform_conv3d = _deform_conv3d_module.deform_conv3d

filename = osp.join(osp.dirname(__file__), 'deform_conv3d_grad.so')
_deform_conv3d_grad_module = tf.load_op_library(filename)
"""
Args:
    Forward:NCLHW
    Filter:N'CL'H'W'
    Offset:GL"H"W"L'H'W'3
    Backward:NC*C'L"H"W"
Attrs:
    strides: S
    dilatation_rates: D
    padding: VALID or SAME
        VALID: L"=(L-L'+1)/S P=0
        SAME: L"=ceil(L/S) P=((L"-1)*S+L'-L)/2
Return:
    Forward_grad:NCLHW
    Filter_grad:N'CL'H'W'
    offset_grad:GL"H"W"L'H'W'3
"""
deform_conv3d_grad = _deform_conv3d_grad_module.deform_conv3d_grad


@ops.RegisterGradient("DeformConv3d")
def _deform_conv3d_grad(op, grad):
    """The gradients for `deform_conv3d`.
    Args:
      op: The `deform_conv` `Operation` that we are differentiating, which we can use
        to find the inputs and outputs of the original op.
      grad: Gradient with respect to the output of the `roi_pool` op.
    Returns:
      Gradients with respect to the input of `zero_out`.
    """
    data = op.inputs[0]
    filter = op.inputs[1]
    offset = op.inputs[2]

    strides = op.get_attr('strides')
    rates = op.get_attr('dilatation_rates')
    padding = op.get_attr('padding')

    # compute gradient
    data_grad = deform_conv3d_grad(data, filter, offset, grad, strides=strides,
                                   dilatation_rates=rates, padding=padding)

    return data_grad
