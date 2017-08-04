import tensorflow as tf

deformable_conv3d_video2col_module = tf.load_op_library('./deformable_conv3d_video2col.so')
with tf.Session(''):
    result = deformable_conv3d_video2col_module.deformable_conv3d_video2col \
        ([[[[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[9, 8, 7], [6, 5, 4], [3, 2, 1]],
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]],  # input

         [[[[1, 2], [1, 2]],
           [[2, 1], [2, 1]]]],  # filter

         [[[[[0, 0], [0, 0]],
            [[0, 0], [0, 0]]],
           [[[0, 0], [0, 0]],
            [[0, 0], [0, 0]]]]]  # offset

         , strides=[1, 1, 1]).eval()
    print(result)

    result1 = deformable_conv3d_video2col_module.deformable_conv3d_video2col \
        ([[[[[1, 2], [4, 5]],
            [[9, 8], [6, 5]]]]],  # input nclhw

         [[[[1, 1], [1, 1]],
           [[1, 1], [1, 1]]]],  # filter

         [[[[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]]]  # offset
         ).eval()
    print(result1)
