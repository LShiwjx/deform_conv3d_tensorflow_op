# Modify from https://github.com/soumith/convnet-benchmarks/blob/master/tensorflow/benchmark_alexnet.py

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from datetime import datetime
import time
import math
import tensorflow as tf
import deform_conv3d_op as deform_conv_op

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 8,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('num_batches', 100,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('forward_only', False,
                            """Only run the forward pass.""")
tf.app.flags.DEFINE_boolean('forward_backward_only', False,
                            """Only run the forward-forward pass.""")
tf.app.flags.DEFINE_string('data_format', 'NCHW',
                           """The data format for Convnet operations.
                           Can be either NHWC or NCHW.
                           """)

parameters = []
timing_entries = []


def time_tensorflow_run(session, target, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    if not isinstance(target, list):
        target = [target]
    target_op = tf.group(*target)
    for i in range(FLAGS.num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target_op)
        duration = time.time() - start_time
        if i > num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f' %
                      (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration
    mn = total_duration / FLAGS.num_batches
    vr = total_duration_squared / FLAGS.num_batches - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
          (datetime.now(), info_string, FLAGS.num_batches, mn, sd))


def run_benchmark():
    global parameters
    timing_entries = []
    with tf.Graph().as_default():
        # Generate some dummy images.
        image_size = 120
        image_channel = 3
        video_size = 3
        kernel_length = 3
        kernel_height = 3
        kernel_width = 3
        out_length = 3
        out_height = 120
        out_width = 120
        kernel_num = 2
        offset_group = 1
        # Note that our padding definition is slightly different the cuda-convnet.
        # In order to force the model to start with the same activations sizes,
        # we add 3 to the image_size and employ VALID padding above.
        image_shape = [FLAGS.batch_size, image_channel, video_size, image_size, image_size]
        offset_shape = [FLAGS.batch_size, offset_group, out_length, out_height, out_width, kernel_length, kernel_height,
                        kernel_width, 3]
        kernel_shape = [kernel_num, image_channel, kernel_length, kernel_height, kernel_width]
        images = tf.Variable(tf.random_normal(image_shape,
                                              dtype=tf.float32,
                                              stddev=1e-1))
        offset = tf.Variable(tf.random_normal(offset_shape,
                                              dtype=tf.float32,
                                              stddev=1e-1))
        kernel = tf.Variable(tf.random_normal(kernel_shape,
                                              dtype=tf.float32,
                                              stddev=1e-1))
        parameters = [kernel]

        last_layer = deform_conv_op.deform_conv3d(images, kernel, offset, padding='SAME')

        # 0.006
        # img_shape = [FLAGS.batch_size, video_size, image_size, image_size, image_channel]
        # ker_shape = [kernel_length, kernel_height, kernel_width, image_channel, image_channel * kernel_num]
        # img = tf.Variable(tf.random_normal(img_shape))
        # ker = tf.Variable(tf.random_normal(ker_shape))
        # parameters = [ker]
        # last_layer = tf.nn.conv3d(img, ker, [1, 1, 1, 1, 1], 'SAME')

        # Build an initialization operation.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph.
        sess = tf.Session()
        sess.run(init)

        run_forward = True
        run_forward_backward = True
        if FLAGS.forward_only and FLAGS.forward_backward_only:
            raise ValueError("Cannot specify --forward_only and "
                             "--forward_backward_only at the same time.")
        if FLAGS.forward_only:
            run_forward_backward = False
        elif FLAGS.forward_backward_only:
            run_forward = False

        if run_forward:
            # Run the forward benchmark.
            timing_entries.append(time_tensorflow_run(sess, last_layer, "Forward"))

        if run_forward_backward:
            # Add a simple objective so we can calculate the backward pass.
            # objective = loss(last_layer, labels)
            loss = lambda x: tf.reduce_sum(x)
            objective = loss(last_layer)
            # Compute the gradient with respect to all the parameters.
            grad = tf.gradients(objective, parameters)
            # Run the backward benchmark.
            timing_entries.append(time_tensorflow_run(sess, grad, "Forward-backward"))

            # if FLAGS.csv_file:
            #     store_data_in_csv(timing_entries)


def main(_):
    run_benchmark()


if __name__ == '__main__':
    tf.app.run()
