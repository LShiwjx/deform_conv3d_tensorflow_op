//
// Created by sl on 8/7/17.
//
#define EIGEN_USE_THREADS
#define EIGEN_USE_GPU

#include "deform_conv3d_grad.h"
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
    REGISTER_OP("DeformConv3dGrad")
            .Input("input: T")
            .Input("filter: T")
            .Input("offset: T")
            .Input("residual: T")
            .Output("input_grad: T")
            .Output("filter_grad: T")
            .Output("offset_grad: T")
            .Attr("T: {float, double}")
            .Attr("strides: list(int)= [1,1,1]")
            .Attr("dilatation_rates: list(int)= [1,1,1]")
            .Attr("padding: {'SAME', 'VALID'} = 'VALID'")
            .SetShapeFn([](InferenceContext *c) {
                c->set_output(0, c->input(0));
                c->set_output(1, c->input(1));
                c->set_output(2, c->input(2));

                return Status::OK();
            });
}

//CPU version
template<typename T>
struct setZero<CPUDevice, T> {
    void operator()(const CPUDevice &d, const int n, T *result_data) {
        cout << "using cpu" << endl;
    }

};
template<typename T>
struct DeformConv3dGradFunctor<CPUDevice, T> {
    void operator()(
            const CPUDevice &d,
            const T *data_input, const T *data_filter, const T *data_offset, const T *data_residual,
            const vector<int64> &input_shape, const vector<int64> &filter_shape,
            const vector<int64> &offset_shape, const vector<int64> &residual_shape,
            const vector<int64> &pad, const vector<int64> &stride, const vector<int64> &dilation,
            T *input_grad_ptr, T *filter_grad_ptr, T *offset_grad_ptr) {
        cout << "using cpu" << endl;
    };
};

template<typename Device, typename T>
class DeformConv3dGradOp : public OpKernel {
public:
    explicit DeformConv3dGradOp(OpKernelConstruction *context) :
            OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("strides", &strides));
        OP_REQUIRES(context, strides.size() == 3,
                    errors::InvalidArgument("strides too large"));
        OP_REQUIRES_OK(context, context->GetAttr("dilatation_rates", &dilatation_rates));
        OP_REQUIRES(context, dilatation_rates.size() == 3,
                    errors::InvalidArgument("dilatation_rates too large"));
        OP_REQUIRES_OK(context, context->GetAttr("padding", &padding));
    }

    void Compute(OpKernelContext *context) override {
        //get the input
        const Tensor &input = context->input(0);
        const Tensor &filter = context->input(1);
        const Tensor &offset = context->input(2);
        const Tensor &residual = context->input(3);

        //calculate something
        vector<int64> output_size = {0, 0, 0};
        vector<int64> pads = {0, 0, 0};

        int64 input_batch_size = input.dim_size(0);
        int64 input_channel = input.dim_size(1);
        vector<int64> input_size = {input.dim_size(2), input.dim_size(3), input.dim_size(4)};
        int64 filter_channel = filter.dim_size(0);
        vector<int64> filter_size = {filter.dim_size(1), filter.dim_size(2), filter.dim_size(3)};
        int64 offset_group = offset.dim_size(0);
        vector<int64> offset_size = {offset.dim_size(1), offset.dim_size(2), offset.dim_size(3),
                                     offset.dim_size(4), offset.dim_size(5), offset.dim_size(6)};
        int64 residual_batch_size = residual.dim_size(0);
        int64 residual_channel = residual.dim_size(1);
        vector<int64> residual_size = {residual.dim_size(2), residual.dim_size(3), residual.dim_size(4)};

        //check everything
        OP_REQUIRES(context, offset.dim_size(7) == 3,
                    errors::InvalidArgument("last dim_size of offset should be 3"));
        OP_REQUIRES(context, residual_batch_size == input_batch_size,
                    errors::InvalidArgument("residual_batch_size"));
        OP_REQUIRES(context, residual_channel == filter_channel * input_channel,
                    errors::InvalidArgument("residual_channel"));
        OP_REQUIRES(context, input_channel % offset_group == 0, errors::InvalidArgument("offset group not divided"));
        for (int i = 0; i < 3; ++i) {
            //get the pad and output size
            OP_REQUIRES_OK(context,
                           GetWindowedOutputSize(input_size[i], filter_size[i],
                                                 strides[i], padding, &(output_size[i]), &pads[i]));
            OP_REQUIRES(context, offset_size[i + 3] == filter_size[i],
                        errors::InvalidArgument("offset: ", offset_size[i + 3], " vs filter", filter_size[i]));

            OP_REQUIRES(context, offset_size[i] == output_size[i],
                        errors::InvalidArgument("offset: ", offset_size[i], " vs output: ", output_size[i]));

            OP_REQUIRES(context, residual_size[i] == output_size[i],
                        errors::InvalidArgument("residual: ", residual_size[i], " vs output: ", output_size[i]));

//            OP_REQUIRES(context, input_size[i] % 2 == 1,
//                        errors::InvalidArgument("input: ", input_size[i], " is not singular "));
//            OP_REQUIRES(context, filter_size[i] % 2 == 1,
//                        errors::InvalidArgument("filter: ", filter_size[i], " is not singular "));
        }


        const T *input_ptr = input.template flat<T>().data();
        const T *filter_ptr = filter.template flat<T>().data();
        const T *offset_ptr = offset.template flat<T>().data();
        const T *residual_ptr = residual.template flat<T>().data();

        //allocate the output
        Tensor *input_grad = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape(
                {input_batch_size, input_channel, input_size[0], input_size[1], input_size[2]}), &input_grad));
        T *input_grad_ptr = input_grad->template flat<T>().data();


        Tensor *filter_grad = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape(
                {filter_channel, filter_size[0], filter_size[1], filter_size[2]}), &filter_grad));
        T *filter_grad_ptr = filter_grad->template flat<T>().data();


        Tensor *offset_grad = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape(
                {offset_group, offset_size[0], offset_size[1], offset_size[2], offset_size[3], offset_size[4],
                 offset_size[5], 3}), &offset_grad));
        T *offset_grad_ptr = offset_grad->template flat<T>().data();


        const vector<int64> residual_shape = {residual_batch_size, input_channel * filter_channel,
                                              residual_size[0], residual_size[1], residual_size[2]};
        const vector<int64> input_shape = {input_batch_size, input_channel,
                                           input_size[0], input_size[1], input_size[2]};
        const vector<int64> filter_shape = {filter_channel, filter_size[0], filter_size[1], filter_size[2]};
        const vector<int64> offset_shape = {offset_group, offset_size[0], offset_size[1], offset_size[2],
                                            offset_size[3], offset_size[4], offset_size[5], 3};

        setZero<Device,T>()(context->eigen_device<Device>(),ProdShape(input_shape),input_grad_ptr);
        setZero<Device,T>()(context->eigen_device<Device>(),ProdShape(filter_shape),filter_grad_ptr);
        setZero<Device,T>()(context->eigen_device<Device>(),ProdShape(offset_shape),offset_grad_ptr);
        DeformConv3dGradFunctor<Device, T>()(
                context->eigen_device<Device>(),
                input_ptr, filter_ptr, offset_ptr, residual_ptr,
                input_shape, filter_shape, offset_shape,residual_shape,
                pads, strides, dilatation_rates,
                input_grad_ptr, filter_grad_ptr, offset_grad_ptr
        );

    }


private:
    vector<int64> strides;
    vector<int64> dilatation_rates;
    Padding padding;
};

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("DeformConv3dGrad")                               \
      .Device(DEVICE_GPU)                                        \
      .TypeConstraint<T>("T"),                                   \
      DeformConv3dGradOp<GPUDevice, T>);
REGISTER_GPU(float);
//REGISTER_GPU(int64);
REGISTER_GPU(double);
#endif  // GOOGLE_CUDA

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("DeformConv3dGrad")                               \
      .Device(DEVICE_CPU)                                        \
      .TypeConstraint<T>("T"),                                   \
      DeformConv3dGrad<CPUDevice, T>);
//REGISTER_CPU(float);
//REGISTER_CPU(int64);
//REGISTER_CPU(double);