//
// Created by sl on 8/7/17.
//
#define EIGEN_USE_THREADS
#define EIGEN_USE_GPU

#include "deformable_conv3d_grad.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "vector"

using namespace std;
using namespace tensorflow;
using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

namespace {
    REGISTER_OP("DeformConvBackpropOp")
            .Input("input: T")
            .Input("filter: T")
            .Input("offset: T")
            .Input("out_grad: T")
            .Output("input_grad: T")
            .Output("filter_grad: T")
            .Output("offset_grad: T")
            .Attr("T: {int64, float, double}")
            .Attr("strides: list(int)")
            .Attr("dilatation_rates: list(int)")
            .Attr("deformable_groups: int")
            .Attr("padding: {'SAME', 'VALID'}")
            .SetShapeFn([](InferenceContext *c) {
                //make sure the rank of input is right
                //NCLHW
                ShapeHandle input_shape;
                TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 5, &input_shape));
                //CLHW
                ShapeHandle filter_shape;
                TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &filter_shape));
                //GLHWD3
                ShapeHandle offset_shape;
                TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 6, &offset_shape));
                //GLHWD3
                ShapeHandle out_grad_shape;
                TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 5, &out_grad_shape));

                //get the attributes
                vector<int64> strides, dilatation_rates;
                int64 deformable_groups;
                Padding padding;
                TF_RETURN_IF_ERROR(c->GetAttr("strides", &strides));
                TF_RETURN_IF_ERROR(c->GetAttr("dilatation_rates", &dilatation_rates));
                TF_RETURN_IF_ERROR(c->GetAttr("deformable_groups", &deformable_groups));
                TF_RETURN_IF_ERROR(c->GetAttr("padding", &padding));

                //calcute the output shape lhw
                vector<int64> output_shape(3);
                vector<int64> pads(3);
                int64 batch_size = c->Value(c->Dim(input_shape, 0));
                int64 input_channels = c->Value(c->Dim(input_shape, 1));
                int64 filter_channels = c->Value(c->Dim(filter_shape, 0));
                int64 volume_filter = c->Value(c->Dim(filter_shape, 1)) *
                                      c->Value(c->Dim(filter_shape, 2)) * c->Value(c->Dim(filter_shape, 3));
                for (int i = 0; i < 3; ++i) {
                    TF_RETURN_IF_ERROR(GetWindowedOutputSize(
                            c->Value(c->Dim(input_shape, i + 2)), c->Value(c->Dim(filter_shape, i + 1)), strides[i],
                            padding, &output_shape[i], &pads[i]));
                }

                cout << "col shape: ";
                cout << batch_size << input_channels << output_shape[0] << output_shape[1] << output_shape[2] <<
                 volume_filter << endl;

                //make sure the value of out_grad_shape is right
                if (c->Value(c->Dim(out_grad_shape, 0)) != batch_size) {
                    return errors::InvalidArgument("x batch != y batch");
                }

                if (c->Value(c->Dim(out_grad_shape, 1)) != input_channels * filter_channels) {
                    return errors::InvalidArgument("x*w channels != y channels");
                }

                if (c->Value(c->Dim(out_grad_shape, 3)) != output_shape[0]) {
                    return errors::InvalidArgument("x l != y l");
                }

                if (c->Value(c->Dim(out_grad_shape, 4)) != output_shape[1]) {
                    return errors::InvalidArgument("x h != y h");
                }

                if (c->Value(c->Dim(out_grad_shape, 5)) != output_shape[2]) {
                    return errors::InvalidArgument("x w != y w");
                }

                //check the single value of lhw
                if (c->Value(c->Dim(input_shape, 2)) % 2 != 1 || c->Value(c->Dim(input_shape, 3)) % 2 != 1
                    || c->Value(c->Dim(input_shape, 4)) % 2 != 1) {
                    return errors::InvalidArgument("the input is not singular");
                }
                //test attr deformable groups
                if (c->Value(c->Dim(out_grad_shape, 1)) % deformable_groups != 0) {
                    return errors::InvalidArgument("deformable_groups should be divided by output channels");
                }
                //test offset depth
                if (c->Value(c->Dim(offset_shape, 4)) != volume_filter) {
                    return errors::InvalidArgument("the depth of offset is not right");
                }
                //test offset last dim
                if (c->Value(c->Dim(offset_shape, 5)) != 3) {
                    return errors::InvalidArgument("the last dim of offset is not right");
                }
                //check the offset value of lhw
                if (c->Value(c->Dim(input_shape, 2)) != c->Value(c->Dim(offset_shape, 1))
                    || c->Value(c->Dim(input_shape, 3)) != c->Value(c->Dim(offset_shape, 2))
                    || c->Value(c->Dim(input_shape, 4)) != c->Value(c->Dim(offset_shape, 3))) {
                    return errors::InvalidArgument("the offset is not same as input");
                }

                //set the output shape
                c->set_output(0, c->input(0));
                c->set_output(1, c->input(1));
                c->set_output(2, c->input(2));
                return Status::OK();
            });
}

//CPU version
template<typename T>
struct DeformableConv3dGradFunctor<CPUDevice, T>{
    void operator()(
            const CPUDevice &d,
            const T *data_im, const T *data_filter, const T *data_offset,
            const TensorShape &im_shape, const TensorShape &col_shape, const TensorShape &kernel_shape,
            const vector<int64> &pad, const vector<int64> &stride, const vector<int64> &dilation,
            int deformable_group, T *img_grad_ptr, T *filter_grad_ptr, T *offset_grad_ptr) {
        cout << "using cpu" << endl;
    };
};

template<typename Device, typename T>
class DeformableConv3dGradOp : public OpKernel {
public:
    explicit DeformableConv3dGradOp(OpKernelConstruction *context) :
            OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("strides", &strides));
        OP_REQUIRES(context, strides.size() == 3,
                    errors::InvalidArgument("strides too large"));
        OP_REQUIRES_OK(context, context->GetAttr("dilatation_rates", &dilatation_rates));
        OP_REQUIRES(context, dilatation_rates.size() == 3,
                    errors::InvalidArgument("dilatation_rates too large"));
        OP_REQUIRES_OK(context, context->GetAttr("deformable_groups", &deformable_groups));
        OP_REQUIRES_OK(context, context->GetAttr("padding", &padding));
    }

    void Compute(OpKernelContext *context) override {
        //get the input
        const Tensor &input = context->input(0);
        const TensorShape &input_shape = input.shape();
        const T *input_ptr = input.template flat<T>().data();

        const Tensor &filter = context->input(1);
        const TensorShape &filter_shape = filter.shape();
        const T *filter_ptr = filter.template flat<T>().data();

        const Tensor &offset = context->input(2);
        const TensorShape &offset_shape = offset.shape();
        const T *offset_ptr = offset.template flat<T>().data();

        const Tensor &out_grad = context->input(3);
        const T *out_grad_ptr = out_grad.template flat<T>().data();
        const TensorShape &out_grad_shape = out_grad.shape();

        //calculate something
        int filter_channels = filter.dim_size(0);
        vector<int64> output_shape = {0, 0, 0};
        vector<int64> pads = {0, 0, 0};
        int64 batch_size = input.dim_size(0);
        int64 input_channel = input.dim_size(1);
        int64 filter_channel = filter.dim_size(0);

        //TODO 输入和滤波器都只能是奇数
        //get the pad and output size
        for (int i = 0; i < 3; ++i) {
            OP_REQUIRES_OK(context,
                           GetWindowedOutputSize(input.dim_size(i + 2), filter.dim_size(i + 1),
                                                 strides[i], padding, &(output_shape[i]), &pads[i])
            );
        }
        //col buffer
        const TensorShape col_shape = TensorShape(
                {batch_size, input_channel,
                 output_shape[0], output_shape[1], output_shape[2],
                 ProdShape(filter_shape, 1, filter_shape.dims())});

        //allocate the output
        Tensor *img_grad = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &img_grad));
        T *img_grad_ptr = img_grad->template flat<T>().data();

        Tensor *filter_grad = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(1, filter_shape, &filter_grad));
        T *filter_grad_ptr = filter_grad->template flat<T>().data();

        Tensor *offset_grad = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(2, offset_shape, &offset_grad));
        T *offset_grad_ptr = offset_grad->template flat<T>().data();

        DeformableConv3dGradFunctor<Device, T>()(
                context->eigen_device<Device>(),
                input_ptr, filter_ptr, offset_ptr,
                input_shape, col_shape, filter_shape,
                pads, strides, dilatation_rates, deformable_groups,
                img_grad_ptr, filter_grad_ptr, offset_grad_ptr
        );

    }


private:
    vector<int64> strides;
    vector<int64> dilatation_rates;
    int64 deformable_groups;
    Padding padding;
};

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("DeformableConv3dGrad")                               \
      .Device(DEVICE_GPU)                                        \
      .TypeConstraint<T>("T"),                                   \
      DeformableConv3dGradOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(int64);
REGISTER_GPU(double);
#endif  // GOOGLE_CUDA

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("DeformableConv3dGrad")                               \
      .Device(DEVICE_CPU)                                        \
      .TypeConstraint<T>("T"),                                   \
      DeformableConv3dGrad<CPUDevice, T>);
//REGISTER_CPU(float);
//REGISTER_CPU(int64);
//REGISTER_CPU(double);