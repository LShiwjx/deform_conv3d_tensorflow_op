#ifndef TENSORFLOW_KERNELS_CONV_OPS_im2col_H_
#define TENSORFLOW_KERNELS_CONV_OPS_im2col_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include <cstring>
#include <vector>
using namespace tensorflow;

typedef Eigen::GpuDevice GPUDevice;
typedef Eigen::ThreadPoolDevice CPUDevice;

template<typename Device, typename DType>
struct deformable_im2col {
    void operator()(const Device &d,
                    const DType *data_im, const DType *data_offset,
                    const TensorShape &im_shape, const TensorShape &col_shape, const TensorShape &kernel_shape,
                    const TensorShape &pad, const TensorShape &stride, const TensorShape &dilation,
                    const int deformable_group, DType *data_col);
};
//template <typename Device,typename DType>
//        struct launch_batch_matmul{
//            static void operator()(OpKernelContext *context, const TensorShape &in_x_shape,
//                            const TensorShape &in_y_shape,
//                            const DType *in_x_ptr,
//                            const DType *in_y_ptr, bool adj_x, bool adj_y, DType *out);
//        };



#endif  // TENSORFLOW_KERNELS_CONV_OPS_im2col_H_
