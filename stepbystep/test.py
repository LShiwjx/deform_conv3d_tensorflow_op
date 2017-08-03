import tensorflow as tf
deformable_conv3d_video2col_module = tf.load_op_library('./deformable_conv3d_video2col.so')
with tf.Session(''):
    deformable_conv3d_video2col_module.deformable_conv3d_video2col([[1, 2], [3, 4]]).eval()