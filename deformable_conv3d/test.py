import tensorflow as tf

deformable_conv3d_module = tf.load_op_library('./deformable_conv3d.so')
offset = [[[[[[-0,-0,-0]]*27]*5]*5]*5] #(1,5,5,5,27,3)
filter = [[[[1,0,0]]*3]*3]
with tf.Session(''):
    result = deformable_conv3d_module.deformable_conv3d \
        ([[[[[111., 112, 113, 114, 115], [121, 122, 123, 124, 125], [131, 132, 133, 134, 135], [141, 142, 143, 144, 145], [151, 152, 153, 154, 155]],
            [[211., 212, 213, 214, 215], [221, 222, 223, 224, 225], [231, 232, 233, 234, 235], [241, 242, 243, 244, 245], [251, 252, 253, 254, 255]],
            [[311., 312, 313, 314, 315], [321, 322, 323, 324, 325], [331, 332, 333, 334, 335], [341, 342, 343, 344, 345], [351, 352, 353, 354, 355]],
            [[411., 412, 413, 414, 415], [421, 422, 423, 424, 425], [431, 432, 433, 434, 435], [441, 442, 443, 444, 445], [451, 452, 453, 454, 455]],
            [[511., 512, 513, 514, 515], [521, 522, 523, 524, 525], [531, 532, 533, 534, 535], [541, 542, 543, 544, 545], [551, 552, 553, 554, 555]]]]],  # input nclhw

         [[[[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # h1
           [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # h2
           [[0, 0, 0], [0, 0, 0], [0, 0, 1]]]],  # filter clhw  s=1 p=0 out 3*3*3
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