




////////////////////
void deformable_conv2d(const int n, const float *data_im, const float *data_offset,
            const int height, const int width, const int kernel_h,
            const int kernel_w,
            const int pad_h, const int pad_w,
            const int stride_h, const int stride_w,
            const int dilation_h, const int dilation_w,
            const int channel_per_deformable_group,
            const int height_col, const int width_col, float *data_col);
/////////////////////
double deformable_conv3d_trilinear(const double *bottom_data,
                                   const int data_width_1d, const int data_width_2d,
                                   const int length, const int height, const int width,
                                   double l, double h, double w);


void deformable_conv3d_im2col(const int n, const double *data_im, const double *data_offset,
                              const int height, const int width, const int length,
                              const int kernel_h, const int kernel_w, const int kernel_l,
                              const int pad_h, const int pad_w, const int pad_l,
                              const int stride_h, const int stride_w, const int stride_l,
                              const int dilation_h, const int dilation_w, const int dilation_l,
                              const int channel_per_deformable_group,
                              const int height_col, const int width_col, const int length_col,
                              double *data_col);

template <typename X>
struct add{
    X operator()(X a, X b);
};


