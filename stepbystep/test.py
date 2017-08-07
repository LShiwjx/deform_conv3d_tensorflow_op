import tensorflow as tf

deformable_conv3d_video2col_module = tf.load_op_library('./deformable_conv3d_video2col.so')
with tf.Session(''):
    result = deformable_conv3d_video2col_module.deformable_conv3d_video2col \
        ([[[[[111., 112, 113], [121, 122, 123], [131, 132, 133]],
            [[211, 212, 213], [221, 222, 223], [231, 232, 233]],
            [[311, 312, 313], [321, 322, 323], [331, 332, 333]]]]],  # input nclhw

         [[[[1.]]]],  # filter clhw  s=1 p=0 out 3*3*3

         [[
             [  # l1
                 [[[0.2, 0.5, 0.5]], [[0, 0, 0]], [[0, 0, 0]]],  # h1
                 [[[0, 0, 0]], [[0, 0, 0]], [[0, 0, 0]]],  # h2
                 [[[0, 0, 0]], [[0, 0, 0]], [[0, 0, 0]]]  # h3
             ],
             [  # l2
                 [[[0., 0, 0]], [[0, 0, 0]], [[0, 0, 0]]],
                 [[[0, 0, 0]], [[0, 0, 0]], [[0, 0, 0]]],
                 [[[0, 0, 0]], [[0, 0, 0]], [[0, 0, 0]]]
             ],
             [  # l3
                 [[[0, 0, 0]], [[0, 0, 0]], [[0, 0, 0]]],
                 [[[0, 0, 0]], [[0, 0, 0]], [[0, 0, 0]]],
                 [[[0, 0, 0]], [[0, 0, 0]], [[0, 0, 0]]]
             ],
         ]]  # offset glhwd3 133313
         , strides=[1, 1, 1], padding='SAME').eval()
    print(result)
    #
    # result1 = deformable_conv3d_video2col_module.deformable_conv3d_video2col \
    #     ([[[[[1, 2], [4, 5]],
    #         [[9, 8], [6, 5]]]]],  # input nclhw
    #
    #      [[[[1, 1], [1, 1]],
    #         [[1, 1], [1, 1]]]],  # filter
    #
    #      [[[[[0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0]]]]]  # offset
    #     ).eval()
    # print(result1)
