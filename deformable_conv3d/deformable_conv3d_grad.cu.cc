//
// Created by sl on 8/8/17.
//
#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#define EIGEN_USE_THREADS

#include "deformable_conv3d_grad.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "time.h"

using namespace tensorflow;

//interpolation
template<typename T>
__device__ T Tri_Linear(const T *bottom_data,
                        const int length, const int height, const int width,
                        const double l, const double h, const double w) {
    //length and area
    const int data_width_1d = width;
    const int data_width_2d = height * width;

    //get the cube, the function floor can not be used in template
    int l_low = floor(l);
    int h_low = floor(h);
    int w_low = floor(w);
    int l_high = l_low + 1 > length ? l_low : l_low + 1;
    int h_high = h_low + 1 > height ? h_low : h_low + 1;
    int w_high = w_low + 1 > width ? w_low : w_low + 1;

    //the corner, format is lhw
    T c000 = bottom_data[l_low * data_width_2d + h_low * data_width_1d + w_low];
    T c001 = bottom_data[l_low * data_width_2d + h_low * data_width_1d + w_high];
    T c010 = bottom_data[l_low * data_width_2d + h_high * data_width_1d + w_low];
    T c011 = bottom_data[l_low * data_width_2d + h_high * data_width_1d + w_high];

    T c100 = bottom_data[l_high * data_width_2d + h_low * data_width_1d + w_low];
    T c101 = bottom_data[l_high * data_width_2d + h_low * data_width_1d + w_high];
    T c110 = bottom_data[l_high * data_width_2d + h_high * data_width_1d + w_low];
    T c111 = bottom_data[l_high * data_width_2d + h_high * data_width_1d + w_high];

    //calculate the distance between the point and corner, using 1 to make sure using the low if equal
    T l_width = w - w_low;
    T h_width = 1 - l_width;
    T l_height = h - h_low;
    T h_height = 1 - l_height;
    T l_length = l - l_low;
    T h_length = 1 - l_length;

    //interpolation
    T c00 = c000 * h_width + c001 * l_width;
    T c01 = c010 * h_width + c011 * l_width;
    T c10 = c100 * h_width + c101 * l_width;
    T c11 = c110 * h_width + c111 * l_width;

    T c0 = c00 * h_height + c01 * l_height;
    T c1 = c10 * h_height + c11 * l_height;

    T c = c0 * h_length + c1 * l_length;

    return c;
}

//------------------------------------------------------forward grad---------------------------------------------//
template<typename T>
__global__ void DeformableConv3dInputGradCudaKernel(
        const T *data_backward, const T *data_filter, const T *data_offset,
        const int forward_channel,
        const int batch_size, const int backward_channel,
        const int backward_length, const int backward_height, const int backward_width,
        const int filter_channel, const int filter_length, const int filter_height, const int filter_width,
        const int off_group, const int off_length,
        const int off_height, const int off_width, const int off_depth,
        const int stride_l, const int stride_h, const int stride_w,
        const int pad_l, const int pad_h, const int pad_w,
        const int dilation_l, const int dilation_h, const int dilation_w,
        T *data_grad_forward,
        int num_kernels) {
    CUDA_1D_KERNEL_LOOP(index, num_kernels) {
        //backward
        const int backward_volume = backward_length * backward_height * backward_width;
        //filter
        const int filter_volume = filter_length * filter_height * filter_width;
        //offset
        const int off_volume = off_length * off_height * off_width;

        //forward
        const int forward_length = off_length;
        const int forward_height = off_height;
        const int forward_width = off_width;
        const int forward_volume = off_volume;
        //current position for forward
        const int w_forward = index % forward_width;
        const int h_forward = index / forward_width % forward_height;
        const int l_forward = index / forward_width / forward_height % forward_length;
        const int c_forward = index / forward_width / forward_height / forward_length % forward_channel;
        const int n_forward = index / forward_width / forward_height / forward_length / forward_channel;
        //current position for backward
        const int n_backward = n_forward;
        const int c_backward = c_forward * filter_channel;
        //current position for offset
        const int channel_per_deformable_group = forward_channel / off_group;
        //group index of offset
        const int g_off = c_forward / channel_per_deformable_group;
        //current data ptr for backward img, backward form is NCLHW
        const T *data_backward_base_ptr =
                data_backward + n_backward * backward_channel * backward_volume + c_backward * backward_volume;
        //current data ptr for filter, format is CLHW
        const T *data_filter_base_ptr = data_filter;
        //current data ptr for offset, format GLHWD3
        const T *data_off_base_ptr = data_offset + g_off * off_volume * off_depth * 3;

        //current data ptr for out, format NCLHW
        T *data_grad_forward_base_ptr =
                data_grad_forward + n_forward * forward_channel * forward_volume + c_forward * forward_volume
                + l_forward * forward_height * forward_width + h_forward * forward_width + w_forward;

        //sigma_backward{  sigma_s{w_s * [ |p-q|>1?0: (p==q?1:|p-q|) ] } * backward }= forward
        for (int c = 0; c < filter_channel; ++c) {
            for (int i = 0; i < backward_length; ++i) {
                for (int j = 0; j < backward_height; ++j) {
                    for (int k = 0; k < backward_width; ++k) {
                        const T *data_backward_ptr = data_backward_base_ptr + c * forward_channel * backward_volume
                                                     + i * backward_height * backward_width + j * backward_width + k;
                        T grad = 0;
                        const int curr_forward_l = i * stride_l - pad_l;
                        const int curr_forward_h = j * stride_h - pad_h;
                        const int curr_forward_w = k * stride_w - pad_w;

                        //for each position of filter, get a offset to compare p and q
                        //p = p_forward + p_filter + offset
                        //q is the position of forward in this cuda thread
                        for (int l = 0; l < filter_length; ++l) {
                            for (int m = 0; m < filter_height; ++m) {
                                for (int n = 0; n < filter_width; ++n) {
                                    const T *data_filter_ptr = data_filter_base_ptr + c * filter_volume +
                                                               l * filter_width * filter_height +
                                                               m * filter_width + n;
                                    //off format nclhwd(l'h'w')3
                                    const T *data_off_ptr = data_off_base_ptr +
                                                            curr_forward_l * off_height * off_width * off_depth * 3 +
                                                            curr_forward_h * off_width * off_depth * 3 +
                                                            curr_forward_w * off_depth * 3 +
                                                            l * filter_height * filter_width * 3 +
                                                            m * filter_width * 3 + n * 3;

                                    //position after adding the offset
                                    T p_l = curr_forward_l + l + data_off_ptr[0];
                                    T p_h = curr_forward_h + m + data_off_ptr[1];
                                    T p_w = curr_forward_w + n + data_off_ptr[2];

                                    //abs, can not use abs() because of the cuda

                                    T a_abs = p_l - l_forward > 0 ? p_l - l_forward : l_forward - p_l;
                                    T b_abs = p_h - h_forward > 0 ? p_h - h_forward : h_forward - p_h;
                                    T c_abs = p_w - w_forward > 0 ? p_w - w_forward : w_forward - p_w;

                                    //compare the p with forward in this cuda thread
                                    T a = a_abs >= 1 ? 0 : (a_abs == 0 ? 1 : a_abs);
                                    T b = b_abs >= 1 ? 0 : (b_abs == 0 ? 1 : b_abs);
                                    T c = c_abs >= 1 ? 0 : (c_abs == 0 ? 1 : c_abs);

                                    grad += (*data_filter_ptr) * a * b * c;

                                }
                            }
                        }//filter
                        *data_grad_forward_base_ptr += *data_backward_ptr * grad;
                    }
                }
            }//backward
        }//filter_channel
    }//CUDA
}

//----------------------------------------------------filter grad-------------------------------------------------//
template<typename T>
__global__ void DeformableConv3dFilterGradCudaKernel(
        const T *data_forward, const T *data_backward, const T *data_offset,
        const int forward_channel,
        const int batch_size, const int backward_channel,
        const int backward_length, const int backward_height, const int backward_width,
        const int filter_channel, const int filter_length, const int filter_height, const int filter_width,
        const int off_group, const int off_length,
        const int off_height, const int off_width, const int off_depth,
        const int stride_l, const int stride_h, const int stride_w,
        const int pad_l, const int pad_h, const int pad_w,
        const int dilation_l, const int dilation_h, const int dilation_w,
        T *data_grad_filter,
        int num_kernels) {
    CUDA_1D_KERNEL_LOOP(index, num_kernels) {
        const int backward_volume = backward_length * backward_height * backward_width;
        const int filter_volume = filter_length * filter_height * filter_width;
        const int off_volume = off_length * off_height * off_width;

        //forward
        const int forward_length = off_length;
        const int forward_height = off_height;
        const int forward_width = off_width;
        const int forward_volume = off_volume;
        //current position for gradient out
        const int w_filter = index % filter_width;
        const int h_filter = index / filter_width % filter_height;
        const int l_filter = index / filter_width / filter_height % filter_length;
        const int c_filter = index / filter_width / filter_height / filter_length;

        //current data ptr for filter, format is CLHW
        T *data_grad_filter_ptr =
                data_grad_filter + c_filter * filter_volume + l_filter * filter_height * filter_width +
                h_filter * filter_width + w_filter;


        //sigma_n sigma_c_forward grad_filter_s' = grad_filter_s
        for (int b = 0; b < batch_size; ++b) {
            for (int c = 0; c < forward_channel; ++c) {
                //current position for offset
                const int channel_per_deformable_group = forward_channel / off_group;
                const int g_off = c / channel_per_deformable_group;
                //current data ptr for backward img, backward form is NC(c_forward,c_filter)LHW
                const T *data_backward_base_ptr =
                        data_backward + b * backward_channel * backward_volume +
                        c * filter_channel * backward_volume + c_filter * backward_volume;

                //current data ptr for offset, format GLHWD3
                const T *data_off_base_ptr = data_offset + g_off * off_volume * off_depth * 3;

                //current data ptr for forward, format NCLHW
                const T *data_forward_base_ptr =
                        data_forward + b * forward_channel * forward_volume + c * forward_volume;

                //sigma_o { sigma_q {x_q * G(p, q) ] } } * y_o = grad_filter_s'
                //p = p' + s + off_p's
                //p' = o*stride - pad
                //G(p, q) = |p-q|>=1 ? 0 : (p==q?1:|p-q|)
                for (int i = 0; i < backward_length; ++i) {
                    for (int j = 0; j < backward_height; ++j) {
                        for (int k = 0; k < backward_width; ++k) {
                            //y_o
                            const T *data_backward_ptr =
                                    data_backward_base_ptr + i * backward_height * backward_width
                                    + j * backward_width + k;
                            //p'= o*stride - pad
                            const int curr_forward_l = i * stride_l - pad_l;
                            const int curr_forward_h = j * stride_h - pad_h;
                            const int curr_forward_w = k * stride_w - pad_w;
                            //off_p's
                            const T *data_off_ptr = data_off_base_ptr +
                                                    curr_forward_l * off_height * off_width * off_depth * 3 +
                                                    curr_forward_h * off_width * off_depth * 3 +
                                                    curr_forward_w * off_depth * 3 +
                                                    l_filter * filter_height * filter_width * 3 +
                                                    h_filter * filter_width * 3 + w_filter * 3;
                            //p = p' + s + off_p's
                            T p_l = i + l_filter + data_off_ptr[0];
                            T p_h = j + h_filter + data_off_ptr[1];
                            T p_w = k + w_filter + data_off_ptr[2];
                            T val = 0;
                            //q
                            for (int l = 0; l < forward_length; ++l) {
                                for (int m = 0; m < forward_height; ++m) {
                                    for (int n = 0; n < forward_width; ++n) {
                                        //x_q
                                        const T *data_forward_ptr =
                                                data_forward_base_ptr + l * forward_height * forward_width +
                                                m * forward_width + n;
                                        //abs of p-q, can not use abs() because of the cuda
                                        T a_abs = p_l - l > 0 ? p_l - l : l - p_l;
                                        T b_abs = p_h - m > 0 ? p_h - m : m - p_h;
                                        T c_abs = p_w - n > 0 ? p_w - n : n - p_w;
                                        //G(p-q) = |p-q|>=1 ? 0 : (p==q?1:|p-q|)
                                        T a = a_abs >= 1 ? 0 : (a_abs == 0 ? 1 : a_abs);
                                        T b = b_abs >= 1 ? 0 : (b_abs == 0 ? 1 : b_abs);
                                        T c = c_abs >= 1 ? 0 : (c_abs == 0 ? 1 : c_abs);
                                        val += (*data_forward_ptr) * a * b * c;
                                    }
                                }
                            }//forward
                            // sigma_o @y_o/@w_s * y_o
                            *data_grad_filter_ptr += (*data_backward_ptr) * val;
                        }
                    }
                }//backward

            }//filter_channel
        }//batch
    }//CUDA
}

//--------------------------------------------------offset grad--------------------------------------------------//
template<typename T>
__global__ void DeformableConv3dOffsetGradCudaKernel(
        const T *data_forward, const T *data_backward, const T *data_filter, const T *data_offset,
        const int forward_channel,
        const int batch_size, const int backward_channel,
        const int backward_length, const int backward_height, const int backward_width,
        const int filter_channel, const int filter_length, const int filter_height, const int filter_width,
        const int off_group, const int off_length,
        const int off_height, const int off_width, const int off_depth,
        const int stride_l, const int stride_h, const int stride_w,
        const int pad_l, const int pad_h, const int pad_w,
        const int dilation_l, const int dilation_h, const int dilation_w,
        T *data_grad_offset,
        int num_kernels) {
    CUDA_1D_KERNEL_LOOP(index, num_kernels) {
        //forward
        const int forward_length = off_length;
        const int forward_height = off_height;
        const int forward_width = off_width;
        //volume
        const int backward_volume = backward_length * backward_height * backward_width;
        const int filter_volume = filter_length * filter_height * filter_width;
        const int off_volume = off_length * off_height * off_width;
        const int forward_volume = off_volume;

        //current position for off
        const int w_filter = index / 3 % filter_width;
        const int h_filter = index / 3 / filter_width % filter_height;
        const int l_filter = index / 3 / filter_width / filter_height % filter_length;
        const int w_forward = index / 3 / filter_volume % forward_width;
        const int h_forward = index / 3 / filter_volume / forward_width % forward_height;
        const int l_forward = index / 3 / filter_volume / forward_width / forward_height % forward_length;
        const int g_off = index / 3 / filter_volume / forward_volume;

        //current position for offset
        const int channel_per_deformable_group = forward_channel / off_group;
        //group index of offset
        const int c_forward = g_off * channel_per_deformable_group;

        //current data ptr for offset_grad, format is GLHWD3
        T *data_grad_offset_ptr =
                data_grad_offset + g_off * off_volume * filter_volume * 3 +
                l_forward * forward_height * forward_width * filter_volume * 3 +
                h_forward * forward_width * filter_volume * 3 +
                w_forward * filter_volume * 3 +
                l_filter * filter_height * filter_width * 3 +
                h_filter * filter_width * 3 + w_filter * 3;

        //current data ptr for offset
        const T *data_off_ptr = data_offset + g_off * off_volume * filter_volume * 3 +
                                l_forward * forward_height * forward_width * filter_volume * 3 +
                                h_forward * forward_width * filter_volume * 3 +
                                w_forward * filter_volume * 3 +
                                l_filter * filter_height * filter_width * 3 +
                                h_filter * filter_width * 3 + w_filter * 3;

        //sigma_n,c1,c2 {w_s * sigma_q{ x_q * G'(p,q)} * y_o} = @y/@off_ps
        for (int n = 0; n < batch_size; ++n) {
            for (int c1 = 0; c1 < channel_per_deformable_group; ++c1) {
                for (int c2 = 0; c2 < filter_channel; ++c2) {

                    //w_s  current data ptr for filter, format CLHW
                    const T *data_filter_ptr = data_offset + c2 * filter_volume +
                                               l_filter * filter_height * filter_width + h_filter * filter_width +
                                               w_filter;
                    //current data ptr
                    const T *data_backward_base_ptr =
                            data_backward + n * backward_channel * backward_volume +
                            (c1 + c_forward) * filter_channel * backward_volume + c2 * backward_volume;
                    const T *data_forward_base_ptr =
                            data_forward + n * forward_channel * forward_volume +
                            (c1 + c_forward) * forward_volume;

                    //p' = o*stride - pad
                    //o
                    const int curr_backward_l = (l_forward - l_filter + pad_l) / stride_l;
                    const int curr_backward_h = (h_forward - h_filter + pad_h) / stride_h;
                    const int curr_backward_w = (w_forward - w_filter + pad_w) / stride_w;
                    //y_o
                    const T *data_backward_ptr =
                            data_backward_base_ptr + curr_backward_l * backward_height * backward_width
                            + curr_backward_h * backward_width + curr_backward_w;

                    //sigma_q{ x_q * G'(p,q)} = grad
                    T grad[3] = {0};
                    for (int i = 0; i < forward_length; ++i) {
                        for (int j = 0; j < forward_height; ++j) {
                            for (int k = 0; k < forward_width; ++k) {
                                //p = p' + off_p's position after adding the offset
                                T p_l = l_forward + data_off_ptr[0];
                                T p_h = h_forward + data_off_ptr[1];
                                T p_w = w_forward + data_off_ptr[2];

                                //abs of p-q, can not use abs() because of the cuda
                                T a_abs = p_l - i > 0 ? p_l - i : i - p_l;
                                T b_abs = p_h - j > 0 ? p_h - j : j - p_h;
                                T c_abs = p_w - k > 0 ? p_w - k : k - p_w;

                                //G'(p, q) = |p-q|>=1 or p==q ? 0 : (p>q?1:-1)
                                T a = (a_abs >= 1 || a_abs == 0) ? 0 : (p_l - i > 0 ? 1 : -1);
                                T b = (b_abs >= 1 || b_abs == 0) ? 0 : (p_h - j > 0 ? 1 : -1);
                                T c = (c_abs >= 1 || c_abs == 0) ? 0 : (p_w - k > 0 ? 1 : -1);

                                //x_q
                                const T *data_forward_ptr =
                                        data_forward_base_ptr +
                                        i * forward_height * forward_width + j * forward_width + k;

                                //x_q * G'(p,q)
                                grad[0] += a * (*data_forward_ptr);
                                grad[1] += b * (*data_forward_ptr);
                                grad[2] += c * (*data_forward_ptr);
                            }
                        }
                    }
                    //w_s * grad * y_o
                    *data_grad_offset_ptr += grad[0] * (*data_filter_ptr) * (*data_backward_ptr);
                    *(data_grad_offset_ptr + 1) += grad[1] * (*data_filter_ptr) * (*data_backward_ptr);
                    *(data_grad_offset_ptr + 2) += grad[2] * (*data_filter_ptr) * (*data_backward_ptr);
                }//filter_channel
            }//filter_channel_per_group
        }//n
    }//CUDA
}

template<typename T>
struct DeformableConv3dGradFunctor<GPUDevice, T> {
    void operator()(
            const GPUDevice &d,
            const T *data_forward, const T *data_backward, const T *data_filter, const T *data_offset,
            const TensorShape &backward_shape, const TensorShape &forward_shape,
            const TensorShape &filter_shape, const TensorShape &offset_shape,
            const vector<int64> &pad, const vector<int64> &stride, const vector<int64> &dilation,
            T *forward_grad_ptr, T *filter_grad_ptr, T *offset_grad_ptr) {
        clock_t t0 = clock();
        //the cuda kernel used should be same as output col size.
        int num_kernels_forward = ProdShape(forward_shape);
        int num_kernels_backward = ProdShape(backward_shape);
        int num_kernels_offset = ProdShape(offset_shape);
        //TODO: what is best value
        int block_count = 1024;
        int thread_per_block = 1024;
        DeformableConv3dInputGradCudaKernel<T>
                << < block_count, thread_per_block, 0, d.stream() >> > (
                data_backward, data_filter, data_offset,
                        forward_shape.dim_size(1),
                        backward_shape.dim_size(0), backward_shape.dim_size(1), backward_shape.dim_size(2),
                        backward_shape.dim_size(3), backward_shape.dim_size(4),
                        filter_shape.dim_size(0), filter_shape.dim_size(1), filter_shape.dim_size(2),
                        filter_shape.dim_size(3),
                        offset_shape.dim_size(0), offset_shape.dim_size(1), offset_shape.dim_size(2),
                        offset_shape.dim_size(3), offset_shape.dim_size(4),
                        stride[0], stride[1], stride[2],
                        pad[0], pad[1], pad[2],
                        dilation[0], dilation[1], dilation[2],
                        forward_grad_ptr,
                        num_kernels_forward);
        clock_t t1 = clock();
        cout << "Input grad: " << (t1 - t0) * 1.0 / CLOCKS_PER_SEC * 1000 << std::endl;
        DeformableConv3dFilterGradCudaKernel<T>
                << < block_count, thread_per_block, 0, d.stream() >> > (
                data_forward, data_backward, data_offset,
                        forward_shape.dim_size(1),
                        backward_shape.dim_size(0), backward_shape.dim_size(1), backward_shape.dim_size(2),
                        backward_shape.dim_size(3), backward_shape.dim_size(4),
                        filter_shape.dim_size(0), filter_shape.dim_size(1), filter_shape.dim_size(2),
                        filter_shape.dim_size(3),
                        offset_shape.dim_size(0), offset_shape.dim_size(1), offset_shape.dim_size(2),
                        offset_shape.dim_size(3), offset_shape.dim_size(4),
                        stride[0], stride[1], stride[2],
                        pad[0], pad[1], pad[2],
                        dilation[0], dilation[1], dilation[2],
                        filter_grad_ptr,
                        num_kernels_backward);
        clock_t t2 = clock();
        cout << "Filter grad: " << (t2 - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << std::endl;
        DeformableConv3dOffsetGradCudaKernel<T>
                << < block_count, thread_per_block, 0, d.stream() >> > (
                data_forward, data_backward, data_filter, data_offset,
                        forward_shape.dim_size(1),
                        backward_shape.dim_size(0), backward_shape.dim_size(1), backward_shape.dim_size(2),
                        backward_shape.dim_size(3), backward_shape.dim_size(4),
                        filter_shape.dim_size(0), filter_shape.dim_size(1), filter_shape.dim_size(2),
                        filter_shape.dim_size(3),
                        offset_shape.dim_size(0), offset_shape.dim_size(1), offset_shape.dim_size(2),
                        offset_shape.dim_size(3), offset_shape.dim_size(4),
                        stride[0], stride[1], stride[2],
                        pad[0], pad[1], pad[2],
                        dilation[0], dilation[1], dilation[2],
                        offset_grad_ptr,
                        num_kernels_offset);
        clock_t t3 = clock();
        cout << "Offset grad: " << (t3 - t2) * 1.0 / CLOCKS_PER_SEC * 1000 << std::endl;
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