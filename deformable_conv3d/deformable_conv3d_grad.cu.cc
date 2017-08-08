//
// Created by sl on 8/8/17.
//
#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#define EIGEN_USE_THREADS

#include "deformable_conv3d_grad.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;


template <typename T>
struct DeformableConv3dGradFunctor<GPUDevice, T> {
    void operator()(
            const GPUDevice &d,
            const T *data_im, const T *data_filter, const T *data_offset,
            const TensorShape &im_shape, const TensorShape &col_shape, const TensorShape &kernel_shape,
            const vector<int64> &pad, const vector<int64> &stride, const vector<int64> &dilation,
            int deformable_group, T *img_grad_ptr, T *filter_grad_ptr, T *offset_grad_ptr){

    };
};

// Instantiate functors for the types of OpKernels registered.
typedef Eigen::GpuDevice GPUDevice;
template
struct DeformableConv3dGradFunctor<GPUDevice, float>;
template
struct DeformableConv3dGradFunctor<GPUDevice, int64>;
template
struct DeformableConv3dGradFunctor<GPUDevice, double>;


#endif  // GOOGLE_CUDA