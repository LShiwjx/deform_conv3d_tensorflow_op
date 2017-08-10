#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#define EIGEN_USE_THREADS

#include "deformable_conv3d.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

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


// Define the CUDA kernel.
template<typename T>
__global__ void DeformableConv3dCudaKernel(const int n, const T *data_im, const T *data_offset,
                                           const int channel_in,
                                           const int length_in, const int height_in, const int width_in,
                                           const int channel_filter,
                                           const int filter_l, const int filter_h, const int filter_w,
                                           const int pad_l, const int pad_h, const int pad_w,
                                           const int stride_l, const int stride_h, const int stride_w,
                                           const int dilation_l, const int dilation_h, const int dilation_w,
                                           const int channel_per_deformable_group,
                                           const int channel_col,
                                           const int length_col, const int height_col, const int width_col,
                                           T *data_col, T *data_output, const T *data_filter) {
    //CUDA assignment
    CUDA_1D_KERNEL_LOOP(index, n) {
        //something for output col, the format is  ncv(lhw)d
        const int volume_filter = filter_h * filter_l * filter_w;
        //current conv point
        const int w_col = index % width_col;
        const int h_col = (index / width_col) % height_col;
        const int l_col = ((index / width_col) / height_col) % length_col;
        //current channel
        const int c_col = (((index / width_col) / height_col) / length_col) % channel_col;
        //current batch
        const int n_col = (((index / width_col) / height_col) / length_col) / channel_col;
        const int depth_col = volume_filter;
        const int volume_col = length_col * height_col * width_col;
        const int depth_offset = volume_filter;

        //something for output
        const int n_out = n_col;
        const int volume_out = volume_col;
        const int channel_out = channel_in * channel_filter;

        //something for input img, the format is ncv(lhw), same like up
        const int w_in = w_col * stride_w - pad_w;
        const int h_in = h_col * stride_h - pad_h;
        const int l_in = l_col * stride_l - pad_l;
        const int c_in = c_col;
        const int n_in = n_col;
        const int volume_in = length_in * height_in * width_in;

        //decide which offset params to use
        const int deformable_group_index = c_col / channel_per_deformable_group;


        //current data ptr for output, format is NCLHW
        T *data_output_base_ptr = data_output + n_out * channel_out * volume_out +
                                  l_col * height_col * width_col + h_col * width_col + w_col;

        //current data ptr for col , col form is NCLHWD
        T *data_col_base_ptr =
                data_col + n_col * channel_col * volume_col * depth_col + c_col * volume_col * depth_col +
                l_col * height_col * width_col * depth_col + h_col * width_col * depth_col +
                w_col * depth_col;
        //current data ptr for input img, img form is NCLHW
        const T *data_img_ptr = data_im + n_in * channel_in * volume_in + c_in * volume_in;

        //current data ptr for offset value, off format is GLHWD3
        const T *data_offset_ptr = data_offset + deformable_group_index * volume_col * depth_offset * 3 +
                                   l_col * height_col * width_col * depth_offset * 3 +
                                   h_col * width_col * depth_offset * 3 + w_col * depth_offset * 3;

        const T *data_filter_base_ptr = data_filter;

        T *data_col_ptr = data_col_base_ptr;
        //for every convolution point, calculate the offset value
        for (int j = 0; j < filter_l; j++) {
            for (int k = 0; k < filter_h; k++) {
                for (int l = 0; l < filter_w; l++) {
                    //get the offset position
                    int data_offset_l_ptr = j * filter_h * filter_w + k * filter_w + l;
                    int data_offset_h_ptr = j * filter_h * filter_w + k * filter_w + l + 1;
                    int data_offset_w_ptr = j * filter_h * filter_w + k * filter_w + l + 2;

                    //get the offset
                    T offset_l = data_offset_ptr[data_offset_l_ptr];
                    T offset_h = data_offset_ptr[data_offset_h_ptr];
                    T offset_w = data_offset_ptr[data_offset_w_ptr];

                    //get the value after add offset
                    T l_in_after = l_in + j * dilation_l + offset_l;
                    T h_in_after = h_in + k * dilation_h + offset_h;
                    T w_in_after = w_in + l * dilation_w + offset_w;

                    //the value if current point is out of the origin img.
                    //TODO: can try different methods
                    T val = 0;
                    if (l_in_after >= 0 && h_in_after >= 0 && w_in_after >= 0 && l_in_after <= length_in - 1 &&
                        h_in_after <= height_in - 1 && w_in_after <= width_in - 1) {
                        //interpolation
                        val = Tri_Linear(data_img_ptr, length_in, height_in, width_in,
                                         l_in_after, h_in_after, w_in_after);
                    }
                    //assignment and update for output
                    *data_col_ptr = val;
                    data_col_ptr += 1;
                }
            }
        }

        //do the multiplication
        const T *data_filter_ptr = data_filter_base_ptr;
        T *data_out_ptr = data_output_base_ptr;
        for (int i = 0; i < channel_filter; ++i) {
            data_out_ptr = data_output_base_ptr + i * volume_out;
            data_filter_ptr = data_filter + i * volume_filter;
            T val = 0;
            for (int j = 0; j < filter_l; ++j) {
                for (int k = 0; k < filter_h; ++k) {
                    for (int l = 0; l < filter_w; ++l) {
                        int64 offset = j * filter_h * filter_w + k * filter_w + l;
                        val += data_filter_ptr[offset] * data_col_base_ptr[offset];
                    }
                }
            }
            *data_out_ptr = val;
        }
    }
}

// Define the GPU implementation that launches the CUDA kernel.
template<typename T>
struct DeformableConv3dFunctor<GPUDevice, T> {
    void operator()(const GPUDevice &d,
                    const T *data_im, const T *data_offset,
                    const TensorShape &im_shape, const TensorShape &col_shape, const TensorShape &filter_shape,
                    const vector<int64> &pad, const vector<int64> &stride, const vector<int64> &dilation,
                    int deformable_group, T *data_col, T *data_output, const T *data_filter) {
        //batch size * channel / groups
        int64 channel_per_deformable_group = im_shape.dim_size(1) / deformable_group;
        //the cuda kernel used should be same as output col size.
        int64 num_kernels = ProdShape(col_shape, 0, 5);
        //TODO: what is best value
        int block_count = 5;
        int thread_per_block = 1024;
        DeformableConv3dCudaKernel<T>
                << < block_count, thread_per_block, 0, d.stream() >> > (
                num_kernels, data_im, data_offset,
                        im_shape.dim_size(1), im_shape.dim_size(2), im_shape.dim_size(3), im_shape.dim_size(4),
                        filter_shape.dim_size(0),
                        filter_shape.dim_size(1), filter_shape.dim_size(2), filter_shape.dim_size(3),
                        pad[0], pad[1], pad[2],
                        stride[0], stride[1], stride[2],
                        dilation[0], dilation[1], dilation[2],
                        channel_per_deformable_group,
                        col_shape.dim_size(1), col_shape.dim_size(2), col_shape.dim_size(3), col_shape.dim_size(4),
                        data_col, data_output, data_filter);
    }
};

// Instantiate functors for the types of OpKernels registered.
typedef Eigen::GpuDevice GPUDevice;
template
struct DeformableConv3dFunctor<GPUDevice, float>;
template
struct DeformableConv3dFunctor<GPUDevice, int64>;
template
struct DeformableConv3dFunctor<GPUDevice, double>;

#endif  // GOOGLE_CUDA