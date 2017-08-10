//
// Created by sl on 8/8/17.
//
#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#define EIGEN_USE_THREADS

#include "deformable_conv3d_grad.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;

template<typename T>
__global__ void DeformableConv3dInputGradCudaKernel(
        const T *data_grad_in, const T *data_filter, const T *data_offset,
        const int grad_out_channel,
        const int batch_size, const int grad_in_channel,
        const int grad_in_length, const int grad_in_height, const int grad_in_width,
        const int filter_channel, const int filter_length, const int filter_height, const int filter_width,
        const int off_group, const int off_length,
        const int off_height, const int off_width, const int off_depth,
        const int stride_l, const int stride_h, const int stride_w,
        const int pad_l, const int pad_h, const int pad_w,
        const int dilation_l, const int dilation_h, const int dilation_w,
        T *data_grad_out,
        int num_kernels) {
    CUDA_1D_KERNEL_LOOP(index, num_kernels) {
//        printf("Hello from block %d, thread %d\n%d %d %d", blockIdx.x, threadIdx.x,
//               grad_in_length, off_depth,filter_channel);
        //gradient in
        const int grad_in_volume = grad_in_length * grad_in_height * grad_in_width;
        //filter
        const int filter_volume = filter_length * filter_height * filter_width;
        //offset
        const int off_volume = off_length * off_height * off_width;

        //gradient out
        const int grad_out_length = off_length;
        const int grad_out_height = off_height;
        const int grad_out_width = off_width;
        const int grad_out_volume = off_volume;
        //current position for gradient in
        const int w_grad_in = index % grad_in_width;
        const int h_grad_in = index / grad_in_width % grad_in_height;
        const int l_grad_in = index / grad_in_width / grad_in_height % grad_in_length;
        const int c_grad_in = index / grad_in_width / grad_in_height / grad_in_length % grad_in_channel;
        const int n_grad_in = index / grad_in_width / grad_in_height / grad_in_length / grad_in_channel;
        //current position for gradient out
        const int n_grad_out = n_grad_in;
        const int c_grad_out = c_grad_in / filter_channel;
        const int l_grad_out = l_grad_in * stride_l - pad_l;
        const int h_grad_out = h_grad_in * stride_h - pad_h;
        const int w_grad_out = w_grad_in * stride_w - pad_w;
        //current channel for filter
        const int c_filter = c_grad_in / grad_out_channel;
        //current position for offset
        const int channel_per_deformable_group = grad_out_channel / off_group;
        const int g_off = c_grad_out / channel_per_deformable_group;
        const int l_off = l_grad_out;
        const int h_off = h_grad_out;
        const int w_off = w_grad_out;


//        printf("Hello from block %d, thread %d\n%d %d %d %d %d\n", blockIdx.x, threadIdx.x,
//               batch_size, grad_in_channel,grad_in_length,grad_in_height,grad_in_width);

//        printf("block %d,thread %d:\n grad_in %d %d %d %d %d\n grad_out %d %d %d %d\n filter %d %d %d %d\n"
//                       "offset %d %d %d %d %d\n posion %d %d %d %d\n", blockIdx.x,threadIdx.x,
//        batch_size,grad_in_channel,grad_in_length,grad_in_height,grad_in_width,grad_out_channel,grad_out_length,
//        grad_out_height,grad_out_width,filter_channel,filter_length,filter_height,filter_width,off_group,
//        off_length,off_height,off_width,off_depth,index, l_grad_in,h_grad_in,w_grad_in);

        //current data ptr for grad_in img, grad_in form is NCLHW
        int vv = n_grad_in * grad_in_channel * grad_in_volume + c_grad_in * grad_in_volume
                 + l_grad_in * grad_in_height * grad_in_width + h_grad_in * grad_in_width + w_grad_in;
        printf(" %d ", vv);
        const T *data_grad_in_base_ptr =
                data_grad_in + vv;
        printf(" %d ", *data_grad_in);
        //current data ptr for filter, format is CLHW
        const T *data_filter_base_ptr = data_filter + c_filter * filter_volume;



        //和传入梯度的位置有关
        //current data ptr for offset, format GLHWD3
        const T *data_off_base_ptr = data_offset + g_off * off_volume * off_depth * 3 +
                                     l_off * off_height * off_width * off_depth * 3 +
                                     h_off * off_width * off_depth * 3 +
                                     w_off * off_depth * 3;
//        T t1 = data_filter_base_ptr[0];
//        T t2 = data_filter_base_ptr[1];
//        T t3 = data_filter_base_ptr[2];
//        T t4 = data_filter_base_ptr[3];
//        T t5 = data_filter_base_ptr[4];
//        printf("Hello from block %d, thread %d\n%d %d %d %d %d\n", blockIdx.x, threadIdx.x,
//               t1, t2, t3,t4, t5);
        //current data ptr for out, format NCLHW
        T *data_grad_out_base_ptr =
                data_grad_out + n_grad_out * grad_out_channel * grad_out_volume + c_grad_out * grad_out_volume;


        //sigma_s{w_s * sigma_q[ x_q * (|p-q|>1?0:|p-q|) ] } * grad_in = grad_out
        for (int i = 0; i < filter_length; ++i) {
            for (int j = 0; j < filter_height; ++j) {
                for (int k = 0; k < filter_width; ++k) {
                    const T *data_filter_ptr = data_filter_base_ptr + i * filter_width * filter_height +
                                               j * filter_width + k;
                    //传入梯度的位置加上滤波器的位置
                    const T *data_off_ptr = data_off_base_ptr + i * filter_height * filter_width * 3 +
                                            j * filter_width * 3 + k * 3;

                    T l_ptr = data_off_ptr[0];
                    T h_ptr = data_off_ptr[1];
                    T w_ptr = data_off_ptr[2];

                    T p_l = l_grad_in + i + l_ptr;
                    T p_h = h_grad_in + j + h_ptr;
                    T p_w = w_grad_in + k + w_ptr;

//                    if(threadIdx.x<1)
//                        printf("Hello from block %d, thread %d\n%d %d %d %d %d\n", blockIdx.x, threadIdx.x,
//                               l_ptr, h_ptr,w_ptr,i, j);
//                    printf("Hello from block %d, thread %d\n%f %f %f %f %f\n", blockIdx.x, threadIdx.x,
//                           l_ptr, h_ptr, w_ptr, *data_filter_ptr, *(data_off_ptr));
                    //for every item in grad_out img
                    for (int l = 0; l < grad_out_length; ++l) {
                        for (int m = 0; m < grad_out_height; ++m) {
                            for (int n = 0; n < grad_out_width; ++n) {

                                T grad = 0;
                                T a = abs(p_l - l) >= 1 ? 0 : (p_l == l ? 1 : abs(p_l - l));
                                T b = abs(p_h - m) >= 1 ? 0 : (p_h == m ? 1 : abs(p_h - m));
                                T c = abs(p_w - n) >= 1 ? 0 : (p_w == n ? 1 : abs(p_w - n));

                                grad = (*data_filter_ptr) * a * b * c;
//                                printf(" %d ",grad);
//
                                T *data_grad_out_ptr = data_grad_out_base_ptr + l * grad_out_height * grad_out_width
                                                       + m * grad_out_width + n;
//                                T va=*data_grad_in_base_ptr;
//                                grad = grad*va;
//                                printf(" %d ",grad);
//                                printf(" \n ");
//                                *data_grad_out_ptr = grad;
//                                *data_grad_out_ptr += T(grad * (*data_grad_in_base_ptr));
                            }
                        }
                    }
                }
            }
        }

    }
}

template<typename T>
struct DeformableConv3dGradFunctor<GPUDevice, T> {
    void operator()(
            const GPUDevice &d,
            const T *data_img, const T *data_grad_in, const T *data_filter, const T *data_offset,
            const TensorShape &grad_in_shape, const TensorShape &img_shape,
            const TensorShape &filter_shape, const TensorShape &offset_shape,
            const vector<int64> &pad, const vector<int64> &stride, const vector<int64> &dilation,
            T *img_grad_ptr, T *filter_grad_ptr, T *offset_grad_ptr) {
        //the cuda kernel used should be same as output col size.
        int num_kernels = ProdShape(grad_in_shape);
        //TODO: what is best value
        int block_count = 1024;
        int thread_per_block = 1024;
//        cout<<num_kernels<<grad_in_shape.dim_size(0)<<grad_in_shape.dim_size(1)<<grad_in_shape.dim_size(2);
        DeformableConv3dInputGradCudaKernel<T>
                << < block_count, thread_per_block, 0, d.stream() >> > (
                data_grad_in, data_filter, data_offset,
                        img_shape.dim_size(1),
                        grad_in_shape.dim_size(0), grad_in_shape.dim_size(1), grad_in_shape.dim_size(2),
                        grad_in_shape.dim_size(3), grad_in_shape.dim_size(4),
                        filter_shape.dim_size(0), filter_shape.dim_size(1), filter_shape.dim_size(2),
                        filter_shape.dim_size(3),
                        offset_shape.dim_size(0), offset_shape.dim_size(1), offset_shape.dim_size(2),
                        offset_shape.dim_size(3), offset_shape.dim_size(4),
                        stride[0], stride[1], stride[2],
                        pad[0], pad[1], pad[2],
                        dilation[0], dilation[1], dilation[2],
                        img_grad_ptr,
                        num_kernels);
//        DeformableConv3dFilterGradCudaKernel<T>
//        <<< block_count, thread_per_block, 0, d.stream() >>> (
//        );
//        DeformableConv3dOffsetGradCudaKernel<T>
//        <<< block_count, thread_per_block, 0, d.stream() >>> (
//        );
    }
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