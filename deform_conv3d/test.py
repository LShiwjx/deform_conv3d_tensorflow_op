import tensorflow as tf

deform_conv3d_module = tf.load_op_library('./deform_conv3d.so')
offset = [[[[[[[[[0, 0.5, -0.5]] * 1] * 1] * 1] * 3] * 3] * 3] * 3] * 2
filters = [[[[1]] * 1] * 1]
inputs = [[[[[1.] * 3] * 3] * 3] * 3] * 2
with tf.Session(''):
    result = deform_conv3d_module.deform_conv3d \
        (inputs,
         filters,
         offset
         ).eval()
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

    # [[
    #     [  # l1
    #         [[[0, 0, 0]], [[0, 0, 0]], [[0, 0, 0]]],  # h1
    #         [[[0, 0, 0]], [[0, 0, 0]], [[0, 0, 0]]],  # h2
    #         [[[0, 0, 0]], [[0, 0, 0]], [[0, 0, 0]]]  # h3
    #     ],
    #     [  # l2
    #         [[[0., 0, 0]], [[0, 0, 0]], [[0, 0, 0]]],
    #         [[[0, 0, 0]], [[0, 0, 0]], [[0, 0, 0]]],
    #         [[[0, 0, 0]], [[0, 0, 0]], [[0, 0, 0]]]
    #     ],
    #     [  # l3
    #         [[[0, 0, 0]], [[0, 0, 0]], [[0, 0, 0]]],
    #         [[[0, 0, 0]], [[0, 0, 0]], [[0, 0, 0]]],
    #         [[[0, 0, 0]], [[0, 0, 0]], [[0, 0, 0]]]
    #     ],
    # ]]  # offset glhwd3 133313
