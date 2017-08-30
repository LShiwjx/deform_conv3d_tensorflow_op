//
// Created by sl on 8/8/17.
//
#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#define EIGEN_USE_THREADS

#include "deform_conv3d_grad.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "time.h"

using namespace tensorflow;

//initialize kernel
template<typename T>
__global__ void setZeroKernel(const int n, T *data) {

    CUDA_1D_KERNEL_LOOP(index, n) {
        *(data + index) = T(0);
    }

}

//initialize functor
template<typename T>
struct setZero<GPUDevice, T> {
    void operator()(const GPUDevice &d, const int n, T *result_data) {
        CudaLaunchConfig config = GetCudaLaunchConfig(n, d);
        setZeroKernel<T> << < config.block_count, config.thread_per_block, 0, d.stream() >> > (n, result_data);
    }

};

//cuda kernel for deform
template<typename T>
__global__ void DeformConv3dGradCudaKernel(
        const T *data_input, const T *data_filter, const T *data_offset, const T *data_residual,
        const int batch_size, const int input_channel,
        const int input_length, const int input_height, const int input_width,
        const int filter_num, const int filter_length, const int filter_height, const int filter_width,
        const int off_group,
        const int residual_length, const int residual_height, const int residual_width,
        const int stride_l, const int stride_h, const int stride_w,
        const int pad_l, const int pad_h, const int pad_w,
        const int dilatation_l, const int dilatation_h, const int dilatation_w,
        T *data_grad_input, T *data_grad_filter, T *data_grad_offset,
        Cuda2DLaunchConfig config2d) {
    CUDA_AXIS_KERNEL_LOOP(x, config2d.virtual_thread_count, x) {
        CUDA_AXIS_KERNEL_LOOP(y, config2d.virtual_thread_count, y) { ;
            const int residual_volume = residual_length * residual_height * residual_width;
            const int filter_volume = filter_length * filter_height * filter_width;
            const int input_volume = input_length * input_height * input_width;

            //current conv point for residual, out format N(C*C')L"H"W"
            const int w_residual = x % residual_width;
            const int h_residual = (x / residual_width) % residual_height;
            const int l_residual = ((x / residual_width) / residual_height) % residual_length;
            const int c_in =
                    (((x / residual_width) / residual_height) / residual_length) % input_channel;
            const int n_filter =
                    (((x / residual_width) / residual_height) / residual_length) / input_channel % filter_num;
            const int n_residual =
                    (((x / residual_width) / residual_height) / residual_length) / input_channel / filter_num %
                    batch_size;
            //current filter point c_filter is the same as c_in
            const int w_filter = y % filter_width;
            const int h_filter = y / filter_width % filter_height;
            const int l_filter = y / filter_width / filter_height % filter_length;


            const T *data_residual_ptr =
                    data_residual + n_residual * filter_num * input_channel * residual_volume +
                    n_filter * input_channel * residual_volume +
                    c_in * residual_volume +
                    l_residual * residual_height * residual_width +
                    h_residual * residual_width +
                    w_residual;

            //conv point for input, input format is NCLHW
            const int w_in = w_residual * stride_w - pad_w;
            const int h_in = h_residual * stride_h - pad_h;
            const int l_in = l_residual * stride_l - pad_l;

            //decide which group of offset params to use
            const int channel_per_deformable_group = input_channel * filter_num / off_group;
            const int deformable_group_index = (n_filter * input_channel + c_in) / channel_per_deformable_group;

            //current data ptr for input img, img format is NCLHW
            const int64 input_off = n_residual * input_channel * input_volume + c_in * input_volume;
            const T *data_input_base_ptr = data_input + input_off;
            T *data_grad_input_base_ptr = data_grad_input + input_off;

            //current data ptr for offset value, off format is NGL"H"W"L'H'W'3
            const int64 off_off = n_residual * off_group * residual_volume * filter_volume * 3 +
                                  deformable_group_index * residual_volume * filter_volume * 3 +
                                  l_residual * residual_height * residual_width * filter_volume * 3 +
                                  h_residual * residual_width * filter_volume * 3 +
                                  w_residual * filter_volume * 3 +
                                  l_filter * filter_height * filter_width * 3 +
                                  h_filter * filter_width * 3 + w_filter * 3;
            const T *data_offset_base_ptr = data_offset + off_off;
            T *data_grad_offset_base_ptr = data_grad_offset + off_off;
            //current data ptr for filter value, filter format is N'CL'H'W'
            const int64 filter_off = n_filter * input_channel * filter_volume +
                                     c_in * filter_volume +
                                     l_filter * filter_height * filter_width +
                                     h_filter * filter_width + w_filter;
            const T *data_filter_ptr = data_filter + filter_off;
            T *data_grad_filter_ptr = data_grad_filter + filter_off;


            const int data_width_1d = input_width;
            const int data_width_2d = input_height * input_width;
            //get the value after add offset
            float l_in_after = l_in + l_filter * dilatation_l + data_offset_base_ptr[0];
            float h_in_after = h_in + h_filter * dilatation_h + data_offset_base_ptr[1];
            float w_in_after = w_in + w_filter * dilatation_w + data_offset_base_ptr[2];

            //interpolation
            if (l_in_after >= 0 && h_in_after >= 0 && w_in_after >= 0 && l_in_after <= input_length - 1 &&
                h_in_after <= input_height - 1 && w_in_after <= input_width - 1) {

                //eight point around
                int l_low = floor(l_in_after);
                int h_low = floor(h_in_after);
                int w_low = floor(w_in_after);

                int l_high = l_low == l_in_after ? l_low : l_low + 1;
                int h_high = h_low == h_in_after ? h_low : h_low + 1;
                int w_high = w_low == w_in_after ? w_low : w_low + 1;

                int a000 = l_low * data_width_2d + h_low * data_width_1d + w_low;
                int a001 = l_low * data_width_2d + h_low * data_width_1d + w_high;
                int a010 = l_low * data_width_2d + h_high * data_width_1d + w_low;
                int a011 = l_low * data_width_2d + h_high * data_width_1d + w_high;
                int a100 = l_high * data_width_2d + h_low * data_width_1d + w_low;
                int a101 = l_high * data_width_2d + h_low * data_width_1d + w_high;
                int a110 = l_high * data_width_2d + h_high * data_width_1d + w_low;
                int a111 = l_high * data_width_2d + h_high * data_width_1d + w_high;

                //value of eight point
                T c000 = data_input_base_ptr[a000];
                T c001 = data_input_base_ptr[a001];
                T c010 = data_input_base_ptr[a010];
                T c011 = data_input_base_ptr[a011];

                T c100 = data_input_base_ptr[a100];
                T c101 = data_input_base_ptr[a101];
                T c110 = data_input_base_ptr[a110];
                T c111 = data_input_base_ptr[a111];

                //six distance
                T l_width = w_in_after - w_low;
                T h_width = 1 - l_width;
                T l_height = h_in_after - h_low;
                T h_height = 1 - l_height;
                T l_length = l_in_after - l_low;
                T h_length = 1 - l_length;

                //interpolution
                T c00 = c000 * h_width + c001 * l_width;
                T c01 = c010 * h_width + c011 * l_width;
                T c10 = c100 * h_width + c101 * l_width;
                T c11 = c110 * h_width + c111 * l_width;

                T c0 = c00 * h_height + c01 * l_height;
                T c1 = c10 * h_height + c11 * l_height;

                T val = c0 * h_length + c1 * l_length;

                //grad for input
                CudaAtomicAdd(
                        data_grad_input_base_ptr + a000,
                        *data_filter_ptr * h_length * h_height * h_width *
                        (*data_residual_ptr));
                CudaAtomicAdd(
                        data_grad_input_base_ptr + a001,
                        *data_filter_ptr * h_length * h_height * l_width *
                        (*data_residual_ptr));
                CudaAtomicAdd(
                        data_grad_input_base_ptr + a010,
                        *data_filter_ptr * h_length * l_height * h_width *
                        (*data_residual_ptr));
                CudaAtomicAdd(
                        data_grad_input_base_ptr + a011,
                        *data_filter_ptr * h_length * l_height * l_width *
                        (*data_residual_ptr));

//                printf("%d %d %f %f %f %d\n", threadIdx.y, threadIdx.x,;

                CudaAtomicAdd(
                        data_grad_input_base_ptr + a100,
                        *data_filter_ptr * l_length * h_height * h_width *
                        (*data_residual_ptr));
                CudaAtomicAdd(
                        data_grad_input_base_ptr + a101,
                        *data_filter_ptr * l_length * h_height * l_width *
                        (*data_residual_ptr));
                CudaAtomicAdd(
                        data_grad_input_base_ptr + a110,
                        *data_filter_ptr * l_length * l_height * h_width *
                        (*data_residual_ptr));
                CudaAtomicAdd(
                        data_grad_input_base_ptr + a111,
                        *data_filter_ptr * l_length * l_height * l_width *
                        (*data_residual_ptr));




                //grad for filter
                CudaAtomicAdd(data_grad_filter_ptr, val * (*data_residual_ptr));
                //grad for offset
                //TODO:test the other value for point is not derivable, improve less. Maybe there are some other methods
//                if(l_length==0) CudaAtomicAdd(data_grad_offset_base_ptr + 0,
//                                              (data_offset_base_ptr[0]>=0?-1:1)*(*data_filter_ptr) * (*data_residual_ptr));
//                else
                CudaAtomicAdd(data_grad_offset_base_ptr + 0,
                              (*data_filter_ptr) * (*data_residual_ptr)
                              * (c100 * h_height * h_width + c101 * h_height * l_width +
                                 c110 * l_height * h_width + c111 * l_height * l_width -
                                 c000 * h_height * h_width - c001 * h_height * l_width -
                                 c010 * l_height * h_width - c011 * l_height * l_width));
//                if(l_height==0) CudaAtomicAdd(data_grad_offset_base_ptr + 1,
//                                              (data_offset_base_ptr[1]>=0?-1:1)*(*data_filter_ptr) * (*data_residual_ptr));
//                else
                CudaAtomicAdd(data_grad_offset_base_ptr + 1,
                              (*data_filter_ptr) * (*data_residual_ptr)
                              * (c010 * h_length * h_width + c011 * h_length * l_width +
                                 c110 * l_length * h_width + c111 * l_length * l_width -
                                 c000 * h_length * h_width - c001 * h_length * l_width -
                                 c100 * l_length * h_width - c101 * l_length * l_width));
//                if(l_width==0) CudaAtomicAdd(data_grad_offset_base_ptr + 2,
//                                             (data_offset_base_ptr[2]>=0?-1:1)*(*data_filter_ptr) * (*data_residual_ptr));
//                else
                CudaAtomicAdd(data_grad_offset_base_ptr + 2,
                              (*data_filter_ptr) * (*data_residual_ptr)
                              * (c001 * h_height * h_length + c101 * h_height * l_length +
                                 c011 * l_height * h_length + c111 * l_height * l_length -
                                 c000 * h_height * h_length - c100 * h_height * l_length -
                                 c010 * l_height * h_length - c110 * l_height * l_length));
            }//if
            else {
                //the gradient for points out of img
//                CudaAtomicAdd(data_grad_offset_base_ptr + 0,
//                              (data_offset_base_ptr[0]>=0?-1:1)*(*data_filter_ptr) * (*data_residual_ptr));
//                CudaAtomicAdd(data_grad_offset_base_ptr + 1,
//                              (data_offset_base_ptr[1]>=0?-1:1)*(*data_filter_ptr) * (*data_residual_ptr));
//                CudaAtomicAdd(data_grad_offset_base_ptr + 2,
//                              (data_offset_base_ptr[2]>=0?-1:1)*(*data_filter_ptr) * (*data_residual_ptr));
            }

        }
    }
}

//deform functor
template<typename T>
struct DeformConv3dGradFunctor<GPUDevice, T> {
    void operator()(
            const GPUDevice &d,
            const T *data_input, const T *data_filter, const T *data_offset, const T *data_residual,
            const vector<int64> &input_shape, const vector<int64> &filter_shape,
            const vector<int64> &offset_shape, const vector<int64> &residual_shape,
            const vector<int64> &pad, const vector<int64> &stride, const vector<int64> &dilatation,
            T *input_grad_ptr, T *filter_grad_ptr, T *offset_grad_ptr) {
        //the cuda kernel.
        int volume_residual = ProdShape(residual_shape);
        int volume_filter = ProdShape(filter_shape, 2);
        Cuda2DLaunchConfig config2d = GetCuda2DLaunchConfig(volume_residual, volume_filter, d);

        DeformConv3dGradCudaKernel<T>
                << < config2d.block_count, config2d.thread_per_block, 0, d.stream() >> > (
                data_input, data_filter, data_offset, data_residual,
                        input_shape[0], input_shape[1], input_shape[2], input_shape[3], input_shape[4],
                        filter_shape[0], filter_shape[2], filter_shape[3], filter_shape[4],
                        offset_shape[1],
                        residual_shape[2], residual_shape[3], residual_shape[4],
                        stride[0], stride[1], stride[2],
                        pad[0], pad[1], pad[2],
                        dilatation[0], dilatation[1], dilatation[2],
                        input_grad_ptr, filter_grad_ptr, offset_grad_ptr,
                        config2d);
    }
};

// Instantiate functors for the types of OpKernels registered.
typedef Eigen::GpuDevice GPUDevice;
template
struct DeformConv3dGradFunctor<GPUDevice, float>;
template
struct DeformConv3dGradFunctor<GPUDevice, double>;

template
struct setZero<GPUDevice, float>;
template
struct setZero<GPUDevice, double>;
#endif  // GOOGLE_CUDA