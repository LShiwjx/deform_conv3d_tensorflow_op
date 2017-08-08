//
// Created by sl on 8/7/17.
//

#ifndef DEFORMABE_CONV_DEFORMABLE_CONV3D_GRAD_H
#define DEFORMABE_CONV_DEFORMABLE_CONV3D_GRAD_H

#include "vector"
#include "tensorflow/core/framework/tensor_shape.h"

using namespace std;
using namespace tensorflow;

template<typename Device, typename T>
struct DeformableConv3dGradFunctor {
    void operator()(const Device &d,
                    const T *data_im, const T *data_filter, const T *data_offset,
                    const TensorShape &im_shape, const TensorShape &col_shape, const TensorShape &kernel_shape,
                    const vector<int64> &pad, const vector<int64> &stride, const vector<int64> &dilation,
                    int deformable_group, T *img_grad_ptr, T *filter_grad_ptr, T *offset_grad_ptr);
};


inline int ProdShape(const TensorShape &shape, int start = 0, int end = -1) {
    int res = 1;
    if (end == -1)
        end = shape.dims();
    for (int i = start; i < end; i++) {
        res *= shape.dim_size(i);
    }
    return res;
}
#endif //DEFORMABE_CONV_DEFORMABLE_CONV3D_GRAD_H
