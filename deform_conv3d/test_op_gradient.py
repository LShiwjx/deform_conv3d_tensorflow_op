import tensorflow as tf
import deform_conv3d_op
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test


class OpTest(test.TestCase):
    def test_gradient(self):
        with self.test_session(use_gpu=True):
            image_size = 5
            image_channel = 3
            video_size = 5
            kernel_length = 3
            kernel_height = 3
            kernel_width = 3
            out_length = 3
            out_height = 3
            out_width = 3
            kernel_channel = 2
            offset_group = 1
            batch_size = 2

            image_shape = [batch_size, image_channel, video_size, image_size, image_size]
            offset_shape = [offset_group, out_length, out_height, out_width, kernel_length, kernel_height, kernel_width,
                            3]
            kernel_shape = [kernel_channel, kernel_length, kernel_height, kernel_width]
            out_shape = [batch_size, image_channel * kernel_channel, out_length, out_height, out_width]

            images = random_ops.random_normal(image_shape,
                                              dtype=tf.float32,
                                              stddev=1e-1)
            offset = random_ops.random_normal(offset_shape,
                                              dtype=tf.float32,
                                              stddev=1e-1)
            kernel = random_ops.random_normal(kernel_shape,
                                              dtype=tf.float32,
                                              stddev=1e-1)

            last_layer = deform_conv3d_op.deform_conv3d(images, kernel, offset)

            err = gradient_checker.compute_gradient_error(images,
                                                          image_shape,
                                                          last_layer,
                                                          out_shape)
            print("error: ", err)
            self.assertLess(err, 1e-4)


if __name__ == "__main__":
    test.main()
