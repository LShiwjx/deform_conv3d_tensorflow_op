import tensorflow as tf
import deform_conv3d_op
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test
from tensorflow.python.framework import constant_op


class OpTest(test.TestCase):
    def test_gradient(self):
        with self.test_session(use_gpu=True):
            image_size = 5
            out_size = 5

            image_channel = 1

            video_size = 3
            out_length = 3

            kernel_length = 3
            kernel_height = 3
            kernel_width = 3
            kernel_num = 1

            offset_group = 1
            batch_size = 1

            inputs_shape = [batch_size, image_channel, video_size, image_size, image_size]
            offset_shape = [batch_size, offset_group, out_length, out_size, out_size, kernel_length, kernel_height,
                            kernel_width,
                            3]
            filters_shape = [kernel_num, image_channel, kernel_length, kernel_height, kernel_width]
            out_shape = [batch_size, image_channel * kernel_num, out_length, out_size, out_size]

            # 由于offset在整数附近会使得不可导，计算容易出偏差
            offset = constant_op.constant([[[[[[[[[0.5, 0.5, 0.5]] * kernel_width] * kernel_height] * kernel_length]
                                              * out_size] * out_size] * out_length] * offset_group] * batch_size,
                                          dtype=tf.float32)
            filters = constant_op.constant(
                [[[[[1.] * kernel_width] * kernel_height] * kernel_length] * image_channel] * kernel_num,
                dtype=tf.float32)
            inputs = constant_op.constant(
                [[[[[1.3] * image_size] * image_size] * video_size] * image_channel] * batch_size,
                dtype=tf.float32)
            # inputs = tf.random_normal()

            last_layer = deform_conv3d_op.deform_conv3d(inputs, filters, offset, padding='SAME')

            err = gradient_checker.compute_gradient_error([inputs, filters, offset],
                                                          [inputs_shape, filters_shape, offset_shape],
                                                          last_layer,
                                                          out_shape)
            self.assertLess(err, 1e-3)


if __name__ == "__main__":
    test.main()
