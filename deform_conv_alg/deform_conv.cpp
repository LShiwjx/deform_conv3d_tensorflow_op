#include "deform_conv.h"
#include <algorithm>


double deformable_conv3d_trilinear(const double *bottom_data,
                                   const int data_width_1d, const int data_width_2d,
                                   const int length, const int height, const int width,
                                   double l, double h, double w) {


    //得到立方体的四个坐标
    int l_low = floor(l);
    int h_low = floor(h);
    int w_low = floor(w);
    int l_high = l_low + 1 > length ? l_low : l_low + 1;
    int h_high = h_low + 1 > height ? h_low : h_low + 1;
    int w_high = w_low + 1 > width ? w_low : w_low + 1;
//    int l_high = l_low + 1;
//    int h_high = h_low + 1;
//    int w_high = w_low + 1;
    //数据存储格式为whl，计算各个角点
    double c000 = bottom_data[l_low * data_width_2d + h_low * data_width_1d + w_low];
    double c001 = bottom_data[l_low * data_width_2d + h_low * data_width_1d + w_high];
    double c010 = bottom_data[l_low * data_width_2d + h_high * data_width_1d + w_low];
    double c011 = bottom_data[l_low * data_width_2d + h_high * data_width_1d + w_high];

    double c100 = bottom_data[l_high * data_width_2d + h_low * data_width_1d + w_low];
    double c101 = bottom_data[l_high * data_width_2d + h_low * data_width_1d + w_high];
    double c110 = bottom_data[l_high * data_width_2d + h_high * data_width_1d + w_low];
    double c111 = bottom_data[l_high * data_width_2d + h_high * data_width_1d + w_high];

    //计算角点与初始点的距离
    double l_width = w - w_low;
    double h_width = 1-l_width;//w_high - w;
    double l_height = h - h_low;
    double h_height = 1-l_height;//h_high - h;
    double l_length = l - l_low;
    double h_length = 1-l_length;//l_high - l;

    //逐步插值到中心点
    double c00 = c000 * h_width + c001 * l_width;
    double c01 = c010 * h_width + c011 * l_width;
    double c10 = c100 * h_width + c101 * l_width;
    double c11 = c110 * h_width + c111 * l_width;

    double c0 = c00 * h_height + c01 * l_height;
    double c1 = c10 * h_height + c11 * l_height;

    double c = c0 * h_length + c1 * l_length;

    return c;
}


void deformable_conv3d_im2col(const int n, const double *data_im, const double *data_offset,
                              const int length,const int height, const int width,
                              const int kernel_l,const int kernel_h, const int kernel_w,
                              const int pad_l,const int pad_h, const int pad_w,
                              const int stride_l,const int stride_h, const int stride_w,
                              const int dilation_l,const int dilation_h, const int dilation_w,
                              const int channel_per_deformable_group,
                              const int length_col,const int height_col, const int width_col,
                              double *data_col) {

    const int w_col = n % width_col;
    const int h_col = (n / width_col) % height_col;
    const int l_col = ((n / width_col) / height_col) % length_col;//输出格子在l方向上的位置
    const int c_im = ((n / width_col) / height_col) / length_col;//目前处理的输入图片的通道
    const int c_col = c_im * kernel_h * kernel_l * kernel_w;//输出的通道
    const int deformable_group_index = c_im / channel_per_deformable_group;//使用第几组偏移量

    const int w_in = w_col * stride_w - pad_w;
    const int h_in = h_col * stride_h - pad_h;
    const int l_in = l_col * stride_l - pad_l;//原图像的卷积点

    double *data_col_ptr = data_col + c_col * length_col * height_col * width_col + l_col * height_col + width_col +
                           h_col * width_col + w_col;//输出指针的初始位置，四维张量clhw
    const double *data_img_ptr =
            data_im + c_im * length * height * width + l_in * height * width + h_in * width + w_in;//图像思维张量，clhw
    const double *data_offset_ptr = data_offset +
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


float deformable_conv2d_bilinear(const float *bottom_data, const int data_width,
                                 const int height, const int width, float h, float w) {
    //////////////////////////////////////////////////
    int h_low = floor(h);
    int w_low = floor(w);
    int h_high = h_low + 1;
    int w_high = w_low + 1;
//    int h_high;
//    int w_high;
//    if (h_low >= height - 1) {
//        h_high = h_low = height - 1;
//        h = (float) h_low;
//    } else {
//        h_high = h_low + 1;
//    }

//    if (w_low >= width - 1) {
//        w_high = w_low = width - 1;
//        w = (float) w_low;
//    } else {
//        w_high = w_low + 1;
//    }
    /////////////////////////////////////////找到四个点
    float lh = h - h_low;
    float lw = w - w_low;
    float hh = 1 - lh, hw = 1 - lw;

    float v1 = bottom_data[h_low * data_width + w_low];
    float v2 = bottom_data[h_low * data_width + w_high];
    float v3 = bottom_data[h_high * data_width + w_low];
    float v4 = bottom_data[h_high * data_width + w_high];
    float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

    float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    return val;
}

void deformable_conv2d(const int n, const float *data_im, const float *data_offset,
                       const int height, const int width, const int kernel_h,
                       const int kernel_w,
                       const int pad_h, const int pad_w,
                       const int stride_h, const int stride_w,
                       const int dilation_h, const int dilation_w,
                       const int channel_per_deformable_group,
                       const int height_col, const int width_col, float *data_col) {
    //分配给cuda的线程 只有一个grid 一个线程做一次卷积操作 一个通道
    // index index of output matrix

    const int w_col = n % width_col;//x方向上的初始便宜量
    const int h_col = (n / width_col) % height_col;//y方向的，不算straid
    const int c_im = (n / width_col) / height_col;//属于第几个channel
    const int c_col = c_im * kernel_h * kernel_w;//属于输出的第几个channel，每一张图片的off是2N

    // compute deformable group index
    const int deformable_group_index = c_im / channel_per_deformable_group;//属于第几组

    const int h_in = h_col * stride_h - pad_h;//在原图像的位置，只减1个pad
    const int w_in = w_col * stride_w - pad_w;
    float *data_col_ptr = data_col + (c_col * height_col + h_col) * width_col + w_col;//c*h*w+h_now*w+w_now
    const float *data_im_ptr = data_im + (c_im * height + h_in) * width + w_in;
    const float *data_offset_ptr =
            data_offset + deformable_group_index * 2 * kernel_h * kernel_w * height_col * width_col;
    //deformable group是对imagechannel的缩减，几个channel共用一个group的偏移量，偏移量大小和输出一样

    for (int i = 0; i < kernel_h; ++i) {
        for (int j = 0; j < kernel_w; ++j) {//i*k_w+j<k_w*k_h
            const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
            const int data_offset_w_ptr =
                    ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col + w_col;
            const float offset_h = data_offset_ptr[data_offset_h_ptr];
            const float offset_w = data_offset_ptr[data_offset_w_ptr];
            float val = static_cast<float>(0);
            const float h_im = h_in + i * dilation_h + offset_h;
            const float w_im = w_in + j * dilation_w + offset_w;//偏移后的位置
            if (h_im >= 0 && w_im >= 0 && h_im <= height - 1 && w_im <= width - 1) {//不再图像内取0,原来是<height
                const float map_h = i * dilation_h + offset_h;//偏移量针对h_in来讲
                const float map_w = j * dilation_w + offset_w;
                const int cur_height = height - h_in;//所以要归一化
                const int cur_width = width - w_in;
                val = deformable_conv2d_bilinear(data_im_ptr, width, cur_height, cur_width, map_h, map_w);
            }
            *data_col_ptr = val;//col是out的形状，每个点又由一个向量组成
            data_col_ptr += height_col * width_col;
        }
    }

}
