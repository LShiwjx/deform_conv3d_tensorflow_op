#ifndef TENSORFLOW_KERNELS_CONV_OPS_im2col_gpu_H_
#define TENSORFLOW_KERNELS_CONV_OPS_im2col_gpu_H_

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#define EIGEN_USE_THREADS


#include "deform_conv.h"
#include "deform_conv_util.h"
#include "cuda.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/kernels/bounds_check.h"

//#include "tensorflow/core/platform/stream_executor.h"

using namespace tensorflow;


//-----------------------------------------------展开------------------------------------------------
//TODO：1
// fetch value from bottom_data(1D array), using subscript (h, w)
template<typename DType>
__device__ DType deformable_im2col_bilinear(const DType *bottom_data, const int data_width,
                                            const int height, const int width, DType h, DType w) {
//////////////////////////////////////////////////
    int h_low = floor(h);
    int w_low = floor(w);
    int h_high;
    int w_high;
    if (h_low >= height - 1) {
        h_high = h_low = height - 1;
        h = (DType) h_low;
    } else {
        h_high = h_low + 1;
    }

    if (w_low >= width - 1) {
        w_high = w_low = width - 1;
        w = (DType) w_low;
    } else {
        w_high = w_low + 1;
    }
/////////////////////////////////////////找到四个点
    DType lh = h - h_low;
    DType lw = w - w_low;
    DType hh = 1 - lh, hw = 1 - lw;

    DType v1 = bottom_data[h_low * data_width + w_low];
    DType v2 = bottom_data[h_low * data_width + w_high];
    DType v3 = bottom_data[h_high * data_width + w_low];
    DType v4 = bottom_data[h_high * data_width + w_high];
    DType w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

    DType val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    return val;
}

//TODO：2
template<typename DType>
__global__ void deformable_im2col_gpu_kernel(const int n, const DType *data_im, const DType *data_offset,
                                             const int height, const int width, const int kernel_h,
                                             const int kernel_w,
                                             const int pad_h, const int pad_w,
                                             const int stride_h, const int stride_w,
                                             const int dilation_h, const int dilation_w,
                                             const int channel_per_deformable_group,
                                             const int height_col, const int width_col,
                                             DType *data_col) {
    CUDA_1D_KERNEL_LOOP(index, n) {//分配给cuda的线程 只有一个grid 一个线程做一次卷积操作 一个通道
        // index index of output matrix
        const int w_col = index % width_col;//x方向上的初始便宜量
        const int h_col = (index / width_col) % height_col;//y方向的，不算straid
        const int c_im = (index / width_col) / height_col;//属于第几个channel
        const int c_col = c_im * kernel_h * kernel_w;//属于输出的第几个channel，每一张图片的off是2N

        // compute deformable group index
        const int deformable_group_index = c_im / channel_per_deformable_group;//属于第几组

        const int h_in = h_col * stride_h - pad_h;//在原图像的位置，只减1个pad
        const int w_in = w_col * stride_w - pad_w;
        DType *data_col_ptr = data_col + (c_col * height_col + h_col) * width_col + w_col;//c*h*w+h_now*w+w_now
        const DType *data_im_ptr = data_im + (c_im * height + h_in) * width + w_in;
        const DType *data_offset_ptr =
                data_offset + deformable_group_index * 2 * kernel_h * kernel_w * height_col * width_col;
        //deformable group是对imagechannel的缩减，几个channel共用一个group的偏移量，偏移量大小和输出一样

        for (int i = 0; i < kernel_h; ++i) {
            for (int j = 0; j < kernel_w; ++j) {//i*k_w+j<k_w*k_h
                const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
                const int data_offset_w_ptr =
                        ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col + w_col;
                const DType offset_h = data_offset_ptr[data_offset_h_ptr];
                const DType offset_w = data_offset_ptr[data_offset_w_ptr];
                DType val = static_cast<DType>(0);
                const DType h_im = h_in + i * dilation_h + offset_h;
                const DType w_im = w_in + j * dilation_w + offset_w;//偏移后的位置
                if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {//不再图像内取0
                    const DType map_h = i * dilation_h + offset_h;//偏移量针对h_in来讲
                    const DType map_w = j * dilation_w + offset_w;
                    const int cur_height = height - h_in;//所以要归一化//TODO：其实永远不会越界
                    const int cur_width = width - w_in;
                    val = deformable_im2col_bilinear(data_im_ptr, width, cur_height, cur_width, map_h, map_w);
                }
                *data_col_ptr = val;
                data_col_ptr += height_col * width_col;
            }
        }
    }
}

//TODO：3
template<typename DType>
struct deformable_im2col<GPUDevice, DType> {
    void operator()(const GPUDevice &d,
                    const DType *data_im, const DType *data_offset,
                    const TensorShape &im_shape, const TensorShape &col_shape, const TensorShape &kernel_shape,
                    const TensorShape &pad, const TensorShape &stride, const TensorShape &dilation,
                    const int deformable_group, DType *data_col) {
        // num_axes should be smaller than block size
        int num_spatial_axes = kernel_shape.dims();//维数，一般是2
        int channel_per_deformable_group = im_shape.dim_size(1) / deformable_group;

        int num_kernels = im_shape.dim_size(1) * ProdShape(col_shape, 1);
        CudaLaunchConfig config = GetCudaLaunchConfig(num_kernels, d);//work_element_count gpu_device
        CHECK_LT(num_spatial_axes, config.thread_per_block);
        switch (num_spatial_axes) {
            case 2:
                deformable_im2col_gpu_kernel<DType> // NOLINT_NEXT_LINE(whitespace/operators)
                        << < config.block_count, config.thread_per_block, 0, d.stream() >> > (
                        num_kernels, data_im, data_offset, im_shape.dim_size(2), im_shape.dim_size(
                                3), kernel_shape.dim_size(0), kernel_shape.dim_size(1),
                                pad.dim_size(0), pad.dim_size(1), stride.dim_size(0), stride.dim_size(
                                1), dilation.dim_size(0), dilation.dim_size(1), channel_per_deformable_group,
                                col_shape.dim_size(1), col_shape.dim_size(2), data_col);
                // MSHADOW_CUDA_POST_KERNEL_CHECK(deformable_im2col_gpu_kernel);
                break;
            default:
                LOG(FATAL) << "im2col_nd_gpu does not support computation with "
                           << num_spatial_axes << " spatial axes";
        }
    }

};

//TODO:4 Instantiate functors for the types of OpKernels registered

template
struct deformable_im2col<GPUDevice, float>;
template
struct deformable_im2col<GPUDevice, double>;

//template
//struct launch_batch_matmul<GPUDevice, float>;
//template
//struct launch_batch_matmul<GPUDevice, double>;

#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_KERNELS_CONV_OPS_im2col_gpu_H_


