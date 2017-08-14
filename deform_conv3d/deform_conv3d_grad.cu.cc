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

//deform kernel
template<typename T>
__global__ void DeformConv3dGradCudaKernel(
        const T *data_input, const T *data_filter, const T *data_offset, const T *data_residual,
        const int batch_size, const int input_channel,
        const int input_length, const int input_height, const int input_width,
        const int filter_channel, const int filter_length, const int filter_height, const int filter_width,
        const int off_group,
        const int residual_length, const int residual_height, const int residual_width,
        const int stride_l, const int stride_h, const int stride_w,
        const int pad_l, const int pad_h, const int pad_w,
        const int dilatation_l, const int dilatation_h, const int dilatation_w,
        T *data_grad_input, T *data_grad_filter, T *data_grad_offset,
        int num_kernels) {
    CUDA_1D_KERNEL_LOOP(index, num_kernels) {
        const int residual_volume = residual_length * residual_height * residual_width;
        const int filter_volume = filter_length * filter_height * filter_width;
        const int input_volume = input_length * input_height * input_width;

        //current conv point for residual, out format N(C*C')L"H"W"
        const int w_residual = index % residual_width;
        const int h_residual = (index / residual_width) % residual_height;
        const int l_residual = ((index / residual_width) / residual_height) % residual_length;
        const int c_filter = (((index / residual_width) / residual_height) / residual_length) % filter_channel;
        const int c_in =
                (((index / residual_width) / residual_height) / residual_length) / filter_channel % input_channel;
        const int n_residual =
                (((index / residual_width) / residual_height) / residual_length) / filter_channel / input_channel;

        const T *data_residual_ptr =
                data_residual + n_residual * input_channel * filter_channel * residual_volume +
                c_in * filter_channel * residual_volume +
                c_filter * residual_volume +
                l_residual * residual_height * residual_width +
                h_residual * residual_width +
                w_residual;

        //conv point for input, input format is NCLHW
        const int w_in = w_residual * stride_w - pad_w;
        const int h_in = h_residual * stride_h - pad_h;
        const int l_in = l_residual * stride_l - pad_l;

        //decide which group of offset params to use
        const int channel_per_deformable_group = input_channel / off_group;
        const int deformable_group_index = c_in / channel_per_deformable_group;

        //current data ptr for input img, img format is NCLHW
        const T *data_input_base_ptr = data_input + n_residual * input_channel * input_volume + c_in * input_volume;
        T *data_grad_input_base_ptr = data_grad_input + n_residual * input_channel * input_volume + c_in * input_volume;

        //current data ptr for offset value, off format is GL"H"W"L'H'W'3
        const T *data_offset_base_ptr = data_offset + deformable_group_index * residual_volume * filter_volume * 3 +
                                        l_residual * residual_height * residual_width * filter_volume * 3 +
                                        h_residual * residual_width * filter_volume * 3 +
                                        w_residual * filter_volume * 3;
        T *data_grad_offset_base_ptr = data_grad_offset + deformable_group_index * residual_volume * filter_volume * 3 +
                                       l_residual * residual_height * residual_width * filter_volume * 3 +
                                       h_residual * residual_width * filter_volume * 3 + w_residual * filter_volume * 3;

        //current data ptr for filter value, off format is C'L'H'W'
        const T *data_filter_base_ptr = data_filter + c_filter * filter_volume;
        T *data_grad_filter_base_ptr = data_grad_filter + c_filter * filter_volume;


        const int data_width_1d = input_width;
        const int data_width_2d = input_height * input_width;

        //for every convolution point, calculate the offset value
        for (int j = 0; j < filter_length; j++) {
            for (int k = 0; k < filter_height; k++) {
                for (int l = 0; l < filter_width; l++) {
                    const int offset_filter = j * filter_height * filter_width + k * filter_width + l;
                    const int offset_off = offset_filter * 3;

                    //get the value after add offset
                    float l_in_after = l_in + j * dilatation_l + data_offset_base_ptr[offset_off];
                    float h_in_after = h_in + k * dilatation_h + data_offset_base_ptr[offset_off + 1];
                    float w_in_after = w_in + l * dilatation_w + data_offset_base_ptr[offset_off + 2];

//                    printf("%f %f %f %d %d %d \n", l_in_after, h_in_after, w_in_after, offset_filter, offset_off, 1);
                    //the value if current point is out of the origin img.
                    //TODO: can try different methods, interpolation and pic?
                    //interpolation
                    if (l_in_after >= 0 && h_in_after >= 0 && w_in_after >= 0 && l_in_after <= input_length - 1 &&
                        h_in_after <= input_height - 1 && w_in_after <= input_width - 1) {

                        //eight point around
                        int l_low = floor(l_in_after);
                        int l_high = l_low == l_in_after ? l_low : l_low + 1;
                        int h_low = floor(h_in_after);
                        int h_high = h_low == h_in_after ? h_low : h_low + 1;
                        int w_low = floor(w_in_after);
                        int w_high = w_low == w_in_after ? w_low : w_low + 1;

                        //value of eight point
                        T c000 = data_input_base_ptr[l_low * data_width_2d + h_low * data_width_1d + w_low];
                        T c001 = data_input_base_ptr[l_low * data_width_2d + h_low * data_width_1d + w_high];
                        T c010 = data_input_base_ptr[l_low * data_width_2d + h_high * data_width_1d + w_low];
                        T c011 = data_input_base_ptr[l_low * data_width_2d + h_high * data_width_1d + w_high];

                        T c100 = data_input_base_ptr[l_high * data_width_2d + h_low * data_width_1d + w_low];
                        T c101 = data_input_base_ptr[l_high * data_width_2d + h_low * data_width_1d + w_high];
                        T c110 = data_input_base_ptr[l_high * data_width_2d + h_high * data_width_1d + w_low];
                        T c111 = data_input_base_ptr[l_high * data_width_2d + h_high * data_width_1d + w_high];

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
                                data_grad_input_base_ptr + l_low * data_width_2d + h_low * data_width_1d + w_low,
                                data_filter_base_ptr[offset_filter] * h_length * h_height * h_width *
                                (*data_residual_ptr));
                        CudaAtomicAdd(
                                data_grad_input_base_ptr + l_low * data_width_2d + h_low * data_width_1d + w_high,
                                data_filter_base_ptr[offset_filter] * h_length * h_height * l_width *
                                (*data_residual_ptr));
                        CudaAtomicAdd(
                                data_grad_input_base_ptr + l_low * data_width_2d + h_high * data_width_1d + w_low,
                                data_filter_base_ptr[offset_filter] * h_length * l_height * h_width *
                                (*data_residual_ptr));
                        CudaAtomicAdd(
                                data_grad_input_base_ptr + l_low * data_width_2d + h_high * data_width_1d + w_high,
                                data_filter_base_ptr[offset_filter] * h_length * l_height * l_width *
                                (*data_residual_ptr));

                        CudaAtomicAdd(
                                data_grad_input_base_ptr + l_high * data_width_2d + h_low * data_width_1d + w_low,
                                data_filter_base_ptr[offset_filter] * l_length * h_height * h_width *
                                (*data_residual_ptr));
                        CudaAtomicAdd(
                                data_grad_input_base_ptr + l_high * data_width_2d + h_low * data_width_1d + w_high,
                                data_filter_base_ptr[offset_filter] * l_length * h_height * l_width *
                                (*data_residual_ptr));
                        CudaAtomicAdd(
                                data_grad_input_base_ptr + l_high * data_width_2d + h_high * data_width_1d + w_low,
                                data_filter_base_ptr[offset_filter] * l_length * l_height * h_width *
                                (*data_residual_ptr));
                        CudaAtomicAdd(
                                data_grad_input_base_ptr + l_high * data_width_2d + h_high * data_width_1d + w_high,
                                data_filter_base_ptr[offset_filter] * l_length * l_height * l_width *
                                (*data_residual_ptr));



                        //grad for filter
                        CudaAtomicAdd(data_grad_filter_base_ptr + offset_filter, val * (*data_residual_ptr));
                        //grad for offset
//                        CudaAtomicAdd(data_grad_offset_base_ptr+offset_off , val
//                                                             * data_filter_base_ptr[offset_filter] * (*data_residual_ptr)
//                                                             * (data_offset_base_ptr[offset_off] > 0 ? 1 :
//                                                                (data_offset_base_ptr[offset_off] == 0 ? 0 : -1)));
//                        CudaAtomicAdd(data_grad_offset_base_ptr+offset_off+1 , val
//                                                             * data_filter_base_ptr[offset_filter] * (*data_residual_ptr)
//                                                             * (data_offset_base_ptr[offset_off + 1] > 0 ? 1 :
//                                                                (data_offset_base_ptr[offset_off + 1] == 0 ? 0 : -1)));
//                        CudaAtomicAdd(data_grad_offset_base_ptr+offset_off+2 , val
//                                                             * data_filter_base_ptr[offset_filter] * (*data_residual_ptr)
//                                                             * (data_offset_base_ptr[offset_off + 2] > 0 ? 1 :
//                                                                (data_offset_base_ptr[offset_off + 2] == 0 ? 0 : -1)));

                        CudaAtomicAdd(data_grad_offset_base_ptr + offset_off,
                                      data_filter_base_ptr[offset_filter] * (*data_residual_ptr)
                                      * (c100 * h_height * h_width + c101 * h_height * l_width +
                                         c110 * l_height * h_width + c111 * l_height * l_width -
                                         c000 * h_height * h_width - c001 * h_height * l_width -
                                         c010 * l_height * h_width - c011 * l_height * l_width));
                        CudaAtomicAdd(data_grad_offset_base_ptr + offset_off + 1,
                                      data_filter_base_ptr[offset_filter] * (*data_residual_ptr)
                                      * (c010 * h_length * h_width + c011 * h_length * l_width +
                                         c110 * l_length * h_width + c111 * l_length * l_width -
                                         c000 * h_length * h_width - c001 * h_length * l_width -
                                         c100 * l_length * h_width - c101 * l_length * l_width));
                        CudaAtomicAdd(data_grad_offset_base_ptr + offset_off + 2,
                                      data_filter_base_ptr[offset_filter] * (*data_residual_ptr)
                                      * (c001 * h_height * h_length + c101 * h_height * l_length +
                                         c011 * l_height * h_length + c111 * l_height * l_length -
                                         c000 * h_height * h_length - c100 * h_height * l_length -
                                         c010 * l_height * h_length - c110 * l_height * l_length));

                    }//if

                }
            }
        }//filter
    }
}

//-------------------------------------------------------offset-------------------------------------------//
template<typename T>
__global__ void DeformConv3dOffsetGradCudaKernel(
        const T *data_input, const T *data_filter, const T *data_offset, const T *data_residual,
        const int batch_size, const int input_channel,
        const int input_length, const int input_height, const int input_width,
        const int filter_channel, const int filter_length, const int filter_height, const int filter_width,
        const int off_group,
        const int residual_length, const int residual_height, const int residual_width,
        const int stride_l, const int stride_h, const int stride_w,
        const int pad_l, const int pad_h, const int pad_w,
        const int dilatation_l, const int dilatation_h, const int dilatation_w,
        T *data_grad_input, T *data_grad_filter, T *data_grad_offset,
        int num_kernels) {
    CUDA_1D_KERNEL_LOOP(index, num_kernels) {
        const int residual_volume = residual_length * residual_height * residual_width;
        const int filter_volume = filter_length * filter_height * filter_width;
        const int input_volume = input_length * input_height * input_width;

        const int index_filter = index % filter_volume;
        const int index_residual = index / filter_volume;
        //current conv point for residual, out format N(C*C')L"H"W"
        const int w_residual = index_residual % residual_width;
        const int h_residual = (index_residual / residual_width) % residual_height;
        const int l_residual = ((index_residual / residual_width) / residual_height) % residual_length;
        const int c_filter = (((index_residual / residual_width) / residual_height) / residual_length) % filter_channel;
        const int c_in =
                (((index_residual / residual_width) / residual_height) / residual_length) / filter_channel %
                input_channel;
        const int n_residual =
                (((index_residual / residual_width) / residual_height) / residual_length) / filter_channel /
                input_channel;
        //current filter point
        const int w_filter = index_filter % filter_width;
        const int h_filter = index_filter / filter_width % filter_height;
        const int l_filter = index_filter / filter_width / filter_height;


        const T *data_residual_ptr =
                data_residual + n_residual * input_channel * filter_channel * residual_volume +
                c_in * filter_channel * residual_volume +
                c_filter * residual_volume +
                l_residual * residual_height * residual_width +
                h_residual * residual_width +
                w_residual;

        //conv point for input, input format is NCLHW
        const int w_in = w_residual * stride_w - pad_w;
        const int h_in = h_residual * stride_h - pad_h;
        const int l_in = l_residual * stride_l - pad_l;

        //decide which group of offset params to use
        const int channel_per_deformable_group = input_channel / off_group;
        const int deformable_group_index = c_in / channel_per_deformable_group;

        //current data ptr for input img, img format is NCLHW
        const T *data_input_base_ptr = data_input + n_residual * input_channel * input_volume + c_in * input_volume;
        T *data_grad_input_base_ptr = data_grad_input + n_residual * input_channel * input_volume + c_in * input_volume;

        //current data ptr for offset value, off format is GL"H"W"L'H'W'3
        const T *data_offset_base_ptr = data_offset + deformable_group_index * residual_volume * filter_volume * 3 +
                                        l_residual * residual_height * residual_width * filter_volume * 3 +
                                        h_residual * residual_width * filter_volume * 3 +
                                        w_residual * filter_volume * 3 +
                                        l_filter * filter_height * filter_width * 3 +
                                        h_filter * filter_width * 3 + w_filter * 3;
        T *data_grad_offset_base_ptr = data_grad_offset + deformable_group_index * residual_volume * filter_volume * 3 +
                                       l_residual * residual_height * residual_width * filter_volume * 3 +
                                       h_residual * residual_width * filter_volume * 3 +
                                       w_residual * filter_volume * 3 +
                                       w_residual * filter_volume * 3 +
                                       l_filter * filter_height * filter_width * 3 +
                                       h_filter * filter_width * 3 + w_filter * 3;
        //current data ptr for filter value, off format is C'L'H'W'
        const T *data_filter_ptr = data_filter + c_filter * filter_volume +
                                   l_filter * filter_height * filter_width +
                                   h_filter * filter_width + w_filter;
        T *data_grad_filter_ptr = data_grad_filter + c_filter * filter_volume +
                                  l_filter * filter_height * filter_width +
                                  h_filter * filter_width + w_filter;


        const int data_width_1d = input_width;
        const int data_width_2d = input_height * input_width;
        //get the value after add offset
        float l_in_after = l_in + l_filter * dilatation_l + data_offset_base_ptr[0];
        float h_in_after = h_in + h_filter * dilatation_h + data_offset_base_ptr[1];
        float w_in_after = w_in + w_filter * dilatation_w + data_offset_base_ptr[2];

        //the value if current point is out of the origin img.
        //TODO: can try different methods, interpolation and pic?
        //interpolation
        if (l_in_after >= 0 && h_in_after >= 0 && w_in_after >= 0 && l_in_after <= input_length - 1 &&
            h_in_after <= input_height - 1 && w_in_after <= input_width - 1) {

            //eight point around
            int l_low = floor(l_in_after);
            int l_high = l_low == l_in_after ? l_low : l_low + 1;
            int h_low = floor(h_in_after);
            int h_high = h_low == h_in_after ? h_low : h_low + 1;
            int w_low = floor(w_in_after);
            int w_high = w_low == w_in_after ? w_low : w_low + 1;

            //value of eight point
            T c000 = data_input_base_ptr[l_low * data_width_2d + h_low * data_width_1d + w_low];
            T c001 = data_input_base_ptr[l_low * data_width_2d + h_low * data_width_1d + w_high];
            T c010 = data_input_base_ptr[l_low * data_width_2d + h_high * data_width_1d + w_low];
            T c011 = data_input_base_ptr[l_low * data_width_2d + h_high * data_width_1d + w_high];

            T c100 = data_input_base_ptr[l_high * data_width_2d + h_low * data_width_1d + w_low];
            T c101 = data_input_base_ptr[l_high * data_width_2d + h_low * data_width_1d + w_high];
            T c110 = data_input_base_ptr[l_high * data_width_2d + h_high * data_width_1d + w_low];
            T c111 = data_input_base_ptr[l_high * data_width_2d + h_high * data_width_1d + w_high];

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
                    data_grad_input_base_ptr + l_low * data_width_2d + h_low * data_width_1d + w_low,
                    *data_filter_ptr * h_length * h_height * h_width *
                    (*data_residual_ptr));
            CudaAtomicAdd(
                    data_grad_input_base_ptr + l_low * data_width_2d + h_low * data_width_1d + w_high,
                    *data_filter_ptr * h_length * h_height * l_width *
                    (*data_residual_ptr));
            CudaAtomicAdd(
                    data_grad_input_base_ptr + l_low * data_width_2d + h_high * data_width_1d + w_low,
                    *data_filter_ptr * h_length * l_height * h_width *
                    (*data_residual_ptr));
            CudaAtomicAdd(
                    data_grad_input_base_ptr + l_low * data_width_2d + h_high * data_width_1d + w_high,
                    *data_filter_ptr * h_length * l_height * l_width *
                    (*data_residual_ptr));

            CudaAtomicAdd(
                    data_grad_input_base_ptr + l_high * data_width_2d + h_low * data_width_1d + w_low,
                    *data_filter_ptr * l_length * h_height * h_width *
                    (*data_residual_ptr));
            CudaAtomicAdd(
                    data_grad_input_base_ptr + l_high * data_width_2d + h_low * data_width_1d + w_high,
                    *data_filter_ptr * l_length * h_height * l_width *
                    (*data_residual_ptr));
            CudaAtomicAdd(
                    data_grad_input_base_ptr + l_high * data_width_2d + h_high * data_width_1d + w_low,
                    *data_filter_ptr * l_length * l_height * h_width *
                    (*data_residual_ptr));
            CudaAtomicAdd(
                    data_grad_input_base_ptr + l_high * data_width_2d + h_high * data_width_1d + w_high,
                    *data_filter_ptr * l_length * l_height * l_width *
                    (*data_residual_ptr));



            //grad for filter
            CudaAtomicAdd(data_grad_filter_ptr, val * (*data_residual_ptr));
            //grad for offset

            CudaAtomicAdd(data_grad_offset_base_ptr + 0,
                          (*data_filter_ptr) * (*data_residual_ptr)
                          * (c100 * h_height * h_width + c101 * h_height * l_width +
                             c110 * l_height * h_width + c111 * l_height * l_width -
                             c000 * h_height * h_width - c001 * h_height * l_width -
                             c010 * l_height * h_width - c011 * l_height * l_width));
            CudaAtomicAdd(data_grad_offset_base_ptr + 1,
                          (*data_filter_ptr) * (*data_residual_ptr)
                          * (c010 * h_length * h_width + c011 * h_length * l_width +
                             c110 * l_length * h_width + c111 * l_length * l_width -
                             c000 * h_length * h_width - c001 * h_length * l_width -
                             c100 * l_length * h_width - c101 * l_length * l_width));
            CudaAtomicAdd(data_grad_offset_base_ptr + 2,
                          (*data_filter_ptr) * (*data_residual_ptr)
                          * (c001 * h_height * h_length + c101 * h_height * l_length +
                             c011 * l_height * h_length + c111 * l_height * l_length -
                             c000 * h_height * h_length - c100 * h_height * l_length -
                             c010 * l_height * h_length - c110 * l_height * l_length));

        }//if

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
        //the cuda kernel used should be same as residual size.
        int num_kernels = ProdShape(residual_shape);
        int num_kernels_offset = ProdShape(residual_shape) * ProdShape(filter_shape);
        CudaLaunchConfig config = GetCudaLaunchConfig(num_kernels, d);
        //TODO:第一种适合小显存，第二种适合大显存
//        clock_t t0 = clock();
        DeformConv3dGradCudaKernel<T>
                << < config.block_count, config.thread_per_block, 0, d.stream() >> > (
                data_input, data_filter, data_offset, data_residual,
                        input_shape[0], input_shape[1], input_shape[2], input_shape[3], input_shape[4],
                        filter_shape[0], filter_shape[1], filter_shape[2], filter_shape[3],
                        offset_shape[0],
                        residual_shape[2], residual_shape[3], residual_shape[4],
                        stride[0], stride[1], stride[2],
                        pad[0], pad[1], pad[2],
                        dilatation[0], dilatation[1], dilatation[2],
                        input_grad_ptr, filter_grad_ptr, offset_grad_ptr,
                        num_kernels);
//        DeformConv3dOffsetGradCudaKernel < T >
//        << < config.block_count, config.thread_per_block, 0, d.stream() >> > (
//                data_input, data_filter, data_offset, data_residual,
//                        input_shape[0], input_shape[1], input_shape[2], input_shape[3], input_shape[4],
//                        filter_shape[0], filter_shape[1], filter_shape[2], filter_shape[3],
//                        offset_shape[0],
//                        residual_shape[2], residual_shape[3], residual_shape[4],
//                        stride[0], stride[1], stride[2],
//                        pad[0], pad[1], pad[2],
//                        dilatation[0], dilatation[1], dilatation[2],
//                        input_grad_ptr, filter_grad_ptr, offset_grad_ptr,
//                        num_kernels_offset);
//        clock_t t1 = clock();
//        cout << "time: " << (t1 - t0) * 1.0 / CLOCKS_PER_SEC * 1000 << std::endl;
//        cout << config.block_count << "   " << config.thread_per_block << endl;
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