#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#define EIGEN_USE_THREADS

#include "deform_conv3d.h"
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
    int l_high = l_low == l ? l_low : l_low + 1;
    int h_high = h_low == h ? h_low : h_low + 1;
    int w_high = w_low == w ? w_low : w_low + 1;

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
__global__ void DeformConv3dCudaKernel(CudaLaunchConfig config,
                                       const T *data_im, const T *data_filter, const T *data_offset,
                                       const int batch_size, const int im_channel,
                                       const int im_l, const int im_h, const int im_w,
                                       const int filter_channel,
                                       const int filter_l, const int filter_h, const int filter_w,
                                       const int offset_group,
                                       const int output_l, const int output_h, const int output_w,
                                       const int pad_l, const int pad_h, const int pad_w,
                                       const int stride_l, const int stride_h, const int stride_w,
                                       const int dilatation_l, const int dilatation_h, const int dilatation_w,
                                       T *data_output) {
    //CUDA assignment
    CUDA_1D_KERNEL_LOOP(index, config.virtual_thread_count) {
        const int volume_filter = filter_w * filter_h * filter_l;
        const int volume_in = im_w * im_h * im_l;
        const int volume_out = output_w * output_h * output_l;
        //current conv point for output, out format N(C*C')L"H"W"
        const int w_out = index % output_w;
        const int h_out = (index / output_w) % output_h;
        const int l_out = ((index / output_w) / output_h) % output_l;
        const int c_in = (((index / output_w) / output_h) / output_l) % im_channel;
        const int c_filter = (((index / output_w) / output_h) / output_l) / im_channel % filter_channel;
        const int n_out = (((index / output_w) / output_h) / output_l) / im_channel / filter_channel;

        //current data ptr for output, format is N(C*C')L"H"W"
        T *data_output_ptr =
                data_output + n_out * im_channel * filter_channel * volume_out +
                c_in * filter_channel * volume_out +
                c_filter * volume_out +
                l_out * output_h * output_w +
                h_out * output_w +
                w_out;

        //conv point for input, input format is NCLHW
        const int w_in = w_out * stride_w - pad_w;
        const int h_in = h_out * stride_h - pad_h;
        const int l_in = l_out * stride_l - pad_l;

        //decide which group of offset params to use
        const int channel_per_deformable_group = im_channel / offset_group;
        const int deformable_group_index = c_in / channel_per_deformable_group;

        //current data ptr for input img, img format is NCLHW
        const T *data_img_base_ptr = data_im + n_out * im_channel * volume_in + c_in * volume_in;

        //current data ptr for offset value, off format is GL"H"W"L'H'W'3
        const T *data_offset_base_ptr = data_offset + n_out * offset_group * volume_out * volume_filter * 3 +
                                        deformable_group_index * volume_out * volume_filter * 3 +
                                        l_out * output_h * output_w * volume_filter * 3 +
                                        h_out * output_w * volume_filter * 3 + w_out * volume_filter * 3;
        //current data ptr for filter value, off format is C'L'H'W'
        const T *data_filter_base_ptr = data_filter + c_filter * volume_filter;


        //result of convolution
        T res = 0;
        //for every convolution point, calculate the offset value
        for (int j = 0; j < filter_l; j++) {
            for (int k = 0; k < filter_h; k++) {
                for (int l = 0; l < filter_w; l++) {
                    int offset = j * filter_h * filter_w + k * filter_w + l;

                    //get the value after add offset
                    T l_in_after = l_in + j * dilatation_l + data_offset_base_ptr[offset * 3];
                    T h_in_after = h_in + k * dilatation_h + data_offset_base_ptr[offset * 3 + 1];
                    T w_in_after = w_in + l * dilatation_w + data_offset_base_ptr[offset * 3 + 2];

                    //the value if current point is out of the origin img.
                    //TODO: can try different methods
                    T val = 0;
                    if (l_in_after >= 0 && h_in_after >= 0 && w_in_after >= 0 && l_in_after <= im_l - 1 &&
                        h_in_after <= im_h - 1 && w_in_after <= im_w - 1) {
                        //interpolation
                        val = Tri_Linear(data_img_base_ptr, im_l, im_h, im_w,
                                         l_in_after, h_in_after, w_in_after);
                    }
                    //convolution
                    res += data_filter_base_ptr[offset] * val;

                }
            }
        }
        *data_output_ptr = res;
    }//CUDA
}

// Define the GPU implementation that launches the CUDA kernel.
template<typename T>
struct DeformConv3dFunctor<GPUDevice, T> {
    void operator()(const GPUDevice &d,
                    const T *data_im, const T *data_filter, const T *data_offset,
                    const vector<int64> &im_shape,
                    const vector<int64> &filter_shape,
                    const vector<int64> &offset_shape,
                    const vector<int64> &output_shape,
                    const vector<int64> &pad, const vector<int64> &stride, const vector<int64> &dilatation,
                    T *data_output) {
        //the cuda kernel used should be same as output col size.
        int num_kernels = ProdShape(output_shape);
        CudaLaunchConfig config = GetCudaLaunchConfig(num_kernels, d);
        DeformConv3dCudaKernel<T>
                << < config.block_count, config.thread_per_block, 0, d.stream() >> > (
                config, data_im, data_filter, data_offset,
                        im_shape[0], im_shape[1], im_shape[2], im_shape[3], im_shape[4],
                        filter_shape[0], filter_shape[1], filter_shape[2], filter_shape[3],
                        offset_shape[1],
                        output_shape[2], output_shape[3], output_shape[4],
                        pad[0], pad[1], pad[2],
                        stride[0], stride[1], stride[2],
                        dilatation[0], dilatation[1], dilatation[2],
                        data_output);
    }
};

// Instantiate functors for the types of OpKernels registered.
typedef Eigen::GpuDevice GPUDevice;
template
struct DeformConv3dFunctor<GPUDevice, float>;
template
struct DeformConv3dFunctor<GPUDevice, double>;

#endif  // GOOGLE_CUDA