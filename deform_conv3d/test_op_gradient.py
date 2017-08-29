import tensorflow as tf
import deform_conv3d_op
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test
from tensorflow.python.framework import constant_op


class OpTest(test.TestCase):
    def test_gradient(self):
        with self.test_session(use_gpu=True):
            image_size = 120
            out_height = 120
            out_width = 120
            image_channel = 3
            video_size = 3
            kernel_length = 3
            kernel_height = 3
            kernel_width = 3
            out_length = 3
            kernel_channel = 2
            offset_group = 3
            batch_size = 3

            inputs_shape = [batch_size, image_channel, video_size, image_size, image_size]
            offset_shape = [batch_size, offset_group, out_length, out_height, out_width, kernel_length, kernel_height,
                            kernel_width,
                            3]
            filters_shape = [kernel_channel, kernel_length, kernel_height, kernel_width]
            out_shape = [batch_size, image_channel * kernel_channel, out_length, out_height, out_width]

            # 由于offset在整数附近会使得不可导，计算容易出偏差
            offset = constant_op.constant([[[[[[[[[0., 0., 0]] * kernel_width] * kernel_height] * kernel_length]
                                              * out_width] * out_height] * out_length] * offset_group] * batch_size,
                                          dtype=tf.float32)
            filters = constant_op.constant([[[[1.] * kernel_width] * kernel_height] * kernel_length] * kernel_channel,
                                           dtype=tf.float32)
            inputs = constant_op.constant(
                [[[[[123.] * image_size] * image_size] * video_size] * image_channel] * batch_size,
                dtype=tf.float32)

            last_layer = deform_conv3d_op.deform_conv3d(inputs, filters, offset, padding='SAME')

            err = gradient_checker.compute_gradient_error([inputs, filters, offset],
                                                          [inputs_shape, filters_shape, offset_shape],
                                                          last_layer,
                                                          out_shape)
            self.assertLess(err, 1e-3)


if __name__ == "__main__":
    test.main()
