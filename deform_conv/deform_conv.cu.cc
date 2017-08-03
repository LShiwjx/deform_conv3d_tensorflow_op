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
template<typename DType>
__device__ DType deformable_conv3d_trilinear(const DType *bottom_data,
                                            const int data_width_1d, const int data_width_2d,
                                            const int length, const int height, const int width,
                                            double l, double h, double w) {


    //得到立方体的四个坐标
    int64 l_low = floor(l);
    int64 h_low = floor(h);
    int64 w_low = floor(w);
    int64 l_high = l_low + 1 > length ? l_low : l_low + 1;
    int64 h_high = h_low + 1 > height ? h_low : h_low + 1;
    int64 w_high = w_low + 1 > width ? w_low : w_low + 1;
//    int l_high = l_low + 1;
//    int h_high = h_low + 1;
//    int w_high = w_low + 1;
    //数据存储格式为whl，计算各个角点
    DType c000 = bottom_data[l_low * data_width_2d + h_low * data_width_1d + w_low];
    DType c001 = bottom_data[l_low * data_width_2d + h_low * data_width_1d + w_high];
    DType c010 = bottom_data[l_low * data_width_2d + h_high * data_width_1d + w_low];
    DType c011 = bottom_data[l_low * data_width_2d + h_high * data_width_1d + w_high];

    DType c100 = bottom_data[l_high * data_width_2d + h_low * data_width_1d + w_low];
    DType c101 = bottom_data[l_high * data_width_2d + h_low * data_width_1d + w_high];
    DType c110 = bottom_data[l_high * data_width_2d + h_high * data_width_1d + w_low];
    DType c111 = bottom_data[l_high * data_width_2d + h_high * data_width_1d + w_high];

    //计算角点与初始点的距离
    DType l_width = w - w_low;
    DType h_width = 1-l_width;//w_high - w;
    DType l_height = h - h_low;
    DType h_height = 1-l_height;//h_high - h;
    DType l_length = l - l_low;
    DType h_length = 1-l_length;//l_high - l;

    //逐步插值到中心点
    DType c00 = c000 * h_width + c001 * l_width;
    DType c01 = c010 * h_width + c011 * l_width;
    DType c10 = c100 * h_width + c101 * l_width;
    DType c11 = c110 * h_width + c111 * l_width;

    DType c0 = c00 * h_height + c01 * l_height;
    DType c1 = c10 * h_height + c11 * l_height;

    DType c = c0 * h_length + c1 * l_length;

    return c;
}

//TODO：2
template<typename DType>
__global__ void deformable_im2col_gpu_kernel(const int n, const DType *data_im, const DType *data_offset,
                                             const int length,const int height, const int width,
                                             const int kernel_l,const int kernel_h, const int kernel_w,
                                             const int pad_l,const int pad_h, const int pad_w,
                                             const int stride_l,const int stride_h, const int stride_w,
                                             const int dilation_l,const int dilation_h, const int dilation_w,
                                             const int channel_per_deformable_group,
                                             const int length_col,const int height_col, const int width_col,
                                             DType *data_col) {
    CUDA_1D_KERNEL_LOOP(index, n) {//分配给cuda的线程 只有一个grid 一个线程做一次卷积操作 一个通道
        const int w_col = index % width_col;
        const int h_col = (index / width_col) % height_col;
        const int l_col = ((index / width_col) / height_col) % length_col;//输出格子在l方向上的位置
        const int c_im = ((index / width_col) / height_col) / length_col;//目前处理的输入图片的通道
        const int c_col = c_im * kernel_h * kernel_l * kernel_w;//输出的通道
        const int deformable_group_index = c_im / channel_per_deformable_group;//使用第几组偏移量

        const int w_in = w_col * stride_w - pad_w;
        const int h_in = h_col * stride_h - pad_h;
        const int l_in = l_col * stride_l - pad_l;//原图像的卷积点

        DType *data_col_ptr = data_col + c_col * length_col * height_col * width_col + l_col * height_col + width_col +
                               h_col * width_col + w_col;//输出指针的初始位置，四维张量clhw
        const DType *data_img_ptr =
                data_im + c_im * length * height * width + l_in * height * width + h_in * width + w_in;//图像思维张量，clhw
        const DType *data_offset_ptr = data_offset +
                                        deformable_group_index * 3 * kernel_h * kernel_l * kernel_w * length_col *
                                        height_col * width_col;//偏移指针的组的起始位置，五维张量

        for (int i = 0; i < kernel_l; ++i) {
            for (int j = 0; j < kernel_h; ++j) {
                for (int k = 0; k < kernel_w; ++k) {
                    const int data_offset_l_ptr =
                            ((3 * (i * kernel_h * kernel_w + j * kernel_w + k) * length_col + l_col) * height_col +
                             h_col) * width_col + w_col;
                    const int data_offset_h_ptr =
                            (((3 * (i * kernel_h * kernel_w + j * kernel_w + k) + 1) * length_col + l_col) * height_col +
                             h_col) * width_col + w_col;
                    const int data_offset_w_ptr =
                            (((3 * (i * kernel_h * kernel_w + j * kernel_w + k) + 2) * length_col + l_col) * height_col +
                             h_col) * width_col + w_col;
                    //组织方式：输出第一次卷积 l....l|h....h|w....w|
                    //           第二次卷积 l....l|h....h|w....w|
                    //....               卷积次数为输出的大小，l的长度为卷积核的大小

                    const double_t offset_l = data_offset_ptr[data_offset_l_ptr];//偏移量
                    const double_t offset_h = data_offset_ptr[data_offset_h_ptr];
                    const double_t offset_w = data_offset_ptr[data_offset_w_ptr];

                    const double_t l_im = l_in + i * dilation_l + offset_l;//偏移后的位置
                    const double_t h_im = h_in + j * dilation_h + offset_h;
                    const double_t w_im = w_in + k * dilation_w + offset_w;

                    double_t val = 0;//偏移后位置如果在图像外，取0 TODO：可以尝试别的取值方式
                    if (l_im >= 0 && h_im >= 0 && w_im >= 0 && l_im <= length - 1 && h_im <= height - 1 &&
                        w_im <= width - 1) {
                        val = deformable_conv3d_trilinear(data_img_ptr, width, width * height, 0, 0, 0, l_im - l_in,
                                                          h_im - h_in, w_im - w_in);//计算铺平的向量的值
                    }
                    *data_col_ptr = val;
                    //输出为x*y*z*|N|,N维铺平的向量长度，即kernel体积
                    data_col_ptr += length_col * height_col * width_col;
                }
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
                    const Tensor &pad, const Tensor &stride, const Tensor &dilation,
                    const int deformable_group, DType *data_col) {
        // 维度 clhw=4
        int64 num_spatial_axes = kernel_shape.dims();
        int64 channel_per_deformable_group = im_shape.dim_size(1) / deformable_group;//TODO

        int64 num_kernels = ProdShape(col_shape, 1) / col_shape.dim_size(5);
        CudaLaunchConfig config = GetCudaLaunchConfig((int) num_kernels, d);//work_element_count gpu_device
        CHECK_LT(num_spatial_axes, config.thread_per_block);
        switch (num_spatial_axes) {
            case 4:
                deformable_im2col_gpu_kernel<DType> // NOLINT_NEXT_LINE(whitespace/operators)
                        << < config.block_count, config.thread_per_block, 0, d.stream() >> > (
                        num_kernels, data_im, data_offset,
                                im_shape.dim_size(2), im_shape.dim_size(3), im_shape.dim_size(4),
                                kernel_shape.dim_size(1), kernel_shape.dim_size(2), kernel_shape.dim_size(3),
                                pad.dim_size(0), pad.dim_size(1), pad.dim_size(2),
                                stride.dim_size(0), stride.dim_size(1), stride.dim_size(2),
                                dilation.dim_size(0), dilation.dim_size(1), dilation.dim_size(2),
                                channel_per_deformable_group,
                                col_shape.dim_size(2), col_shape.dim_size(3), col_shape.dim_size(4),
                                data_col);
            default:return;
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


