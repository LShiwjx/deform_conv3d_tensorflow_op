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
struct DeformConv3dGradFunctor {
    void operator()(const Device &d,
                    const T *data_input,  const T *data_filter, const T *data_offset,const T *data_residual,
                    const vector<int64> &input_shape, const vector<int64> &filter_shape,
                    const vector<int64> &offset_shape,const vector<int64> &residual_shape,
                    const vector<int64> &pad, const vector<int64> &stride, const vector<int64> &dilatation,
                    T *input_grad_ptr, T *filter_grad_ptr, T *offset_grad_ptr);
};

template<typename Device, typename T>
struct setZero {
    void operator()(const Device &d, const int n, T *result_data);

};

inline int64 ProdShape(const vector<int64> &shape, int start = 0, int end = -1) {
    int64 res = 1;
    if (end == -1)
        end = shape.size();
    for (int i = start; i < end; i++) {
        res *= shape[i];
    }
    return res;
}

#endif //DEFORMABE_CONV_DEFORMABLE_CONV3D_GRAD_H
