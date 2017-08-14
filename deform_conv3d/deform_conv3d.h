//
// Created by sl on 8/2/17.
//

#ifndef DEFORMABE_CONV_DEFORMABLE_CONV3D_IM2COL_H
#define DEFORMABE_CONV_DEFORMABLE_CONV3D_IM2COL_H

#include "vector"
#include "tensorflow/core/framework/tensor_shape.h"

using namespace std;
using namespace tensorflow;

template<typename Device, typename T>
struct DeformConv3dFunctor {
    void operator()(const Device &d,
                    const T *data_im, const T *data_filter, const T *data_offset,
                    const vector<int64> &im_shape,
                    const vector<int64> &filter_shape,
                    const vector<int64> &offset_shape,
                    const vector<int64> &output_shape,
                    const vector<int64> &pad, const vector<int64> &stride, const vector<int64> &dilatation,
                    T *data_output);
};

inline int ProdShape(const vector<int64> &shape, int start = 0, int end = -1) {
    int res = 1;
    if (end == -1)
        end = shape.size();
    for (int i = start; i < end; i++) {
        res *= shape[i];
    }
    return res;
}

#endif //DEFORMABE_CONV_DEFORMABLE_CONV3D_IM2COL_H
