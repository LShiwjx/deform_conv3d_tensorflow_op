from __future__ import absolute_import
import tensorflow as tf
import os.path as osp
from tensorflow.python.framework import ops

# python wrapper
filename = osp.join(osp.dirname(__file__), 'deformable_conv3d.so')
_deformable_conv3d_module = tf.load_op_library(filename)
"""
Args:
    Forward:NCLHW
    Filter:CLHW
    Offset:GLHWD3
Attrs:
    strides:
    dilatation_rates:
    padding: VALID or SAME
Return:
    Backward:NCLHW
"""
deformable_conv3d = _deformable_conv3d_module.deformable_conv3d

filename = osp.join(osp.dirname(__file__), 'deformable_conv3d_grad.so')
_deformable_conv3d_grad_module = tf.load_op_library(filename)
"""
Args:
    Forward:NCLHW
    Filter:CLHW
    Offset:GLHWD3
    Backward:NCLHW
Attrs:
    strides:
    dilatation_rates:
    padding: VALID or SAME
Return:
    Forward_grad:NCLHW
    Filter_grad:CLHW
    Backward_grad:NCLHW
"""
deformable_conv3d_grad = _deformable_conv3d_grad_module.deformable_conv3d_grad


@ops.RegisterGradient("DeformableConv3d")
def _deformable_conv3d_grad(op, grad):
    """The gradients for `deformable_conv3d`.
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
    data_grad = deformable_conv3d_grad(data, filter, offset, grad, strides=strides,
                                       dilatation_rates=rates, padding=padding)

    return data_grad
