import tensorflow as tf
import deform_conv3d_op
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test
from tensorflow.python.framework import constant_op


class OpTest(test.TestCase):
    def test_gradient(self):
        with self.test_session(use_gpu=True):
            image_size = 3
            image_channel = 3
            video_size = 3
            kernel_length = 3
            kernel_height = 3
            kernel_width = 3
            out_length = 1
            out_height = 1
            out_width = 1
            kernel_channel = 2
            offset_group = 1
            batch_size = 3

            inputs_shape = [batch_size, image_channel, video_size, image_size, image_size]
            offset_shape = [offset_group, out_length, out_height, out_width, kernel_length, kernel_height, kernel_width,
                            3]
            filters_shape = [kernel_channel, kernel_length, kernel_height, kernel_width]
            out_shape = [batch_size, image_channel * kernel_channel, out_length, out_height, out_width]

            # 由于offset在整数附近会使得不可导，计算容易出偏差
            offset = constant_op.constant([[[[[[[[0.3, 0.2, 0]] * 3] * 3] * 3] * 1] * 1] * 1])
            filters = constant_op.constant([[[[0.5, 0.6, 0.21]] * 3] * 3] * 2)
            inputs = constant_op.constant([[[[[0.4, 0.6, 0.8]] * 3] * 3] * 3] * 3)

            last_layer = deform_conv3d_op.deform_conv3d(inputs, filters, offset)

            err = gradient_checker.compute_gradient_error([inputs, filters, offset],
                                                          [inputs_shape, filters_shape, offset_shape],
                                                          last_layer,
                                                          out_shape)
            print("error: ", err)
            self.assertLess(err, 1e-3)


if __name__ == "__main__":
    test.main()
