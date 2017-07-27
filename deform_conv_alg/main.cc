#include "iostream"
#include "deform_conv.h"
#include <vector>


int main(int argc, char **argv)
{


	using namespace std;
    //----------------------------------test 2d------------------------------------
	int stride = 1,dilation=1,channel_per_deformable_group=1,pad=0;
	int tmp[9]={1,1,3,3,2,2,4,2,2};
	vector<int> im_shape(tmp,tmp+4);
	vector<int> kernel_shape(tmp+4,tmp+6);
	vector<int> col_shape(tmp+6,tmp+9);

	int num_kernels = im_shape[1]*kernel_shape[0]*kernel_shape[1];
	float data_im[9]={1,2,3,4,5,6,7,8,9};
	float data_offset[]={
	                -0,0,0,-1.2,0,0,0,0,//5689
	                -0,0,0,-2.2,0,0,0,0,
	                -0,0,0,0.2,0,0,0,0,
	                -0,0,0,1,0,0,0,0
	        };
	float data_col[16]={0};
    deformable_conv2d(num_kernels-1, data_im, data_offset, im_shape[2], im_shape[3], kernel_shape[0], kernel_shape[1],
    				pad, pad, stride, stride, dilation, dilation, channel_per_deformable_group,
    				col_shape[1], col_shape[2], data_col);


    for(int i=0;i<16;i++) cout<<data_col[i]<<" ";
    cout<<std::endl;
    //----------------------------------test 3d------------------------------------
    double data_im_3d[8]={1,2,3,4,5,6,7,8};
    double data_offset_3d[]={
            0,0,0,0,0,0,0,0,/*l*/   0,0,0,0,0,0,0,0,/*h*/   0,0,0,0,0,0,0,-0.2/*w*/
    };
    double data_col_3d[8]={0};
    deformable_conv3d_im2col(7,data_im_3d,data_offset_3d,2,2,2,1,1,1,0,0,0,1,1,1,1,1,1,1,2,2,2,data_col_3d);
    for(int i=0;i<8;i++) cout<<data_col_3d[i]<<" ";
    cout<<endl;

    //----------------------------------test trilinear-----------------------------------------
    double a[8]={1,1,1,1,1,1,1,1};
    double x = deformable_conv3d_trilinear(&a[0],1,4,1,1,1,0.5,0.5,0.5);
	cout<<x;




	return 0;
}
