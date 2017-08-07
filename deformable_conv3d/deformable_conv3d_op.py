from __future__ import absolute_import
import tensorflow as tf
import os.path as osp

# python wrapper
filename = osp.join(osp.dirname(__file__), 'deformable_conv3d.so')
_deformable_conv3d_module = tf.load_op_library(filename)
deformable_conv3d_op = _deformable_conv3d_module.deformable_conv3d

@ops.RegisterGradient("DeformableConv3dOp")
def _deform_conv_grad(op, grad):
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
    rates = op.get_attr('dilation_rates')
    padding = op.get_attr('padding')
    deformable_groups = op.get_attr('deformable_groups')

    # compute gradient
    data_grad = deformable_conved_grad_op(data, filter, offset, grad, strides,
                                          rates, deformable_groups, padding)

    return data_grad