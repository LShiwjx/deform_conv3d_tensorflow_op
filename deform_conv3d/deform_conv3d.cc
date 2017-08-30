//
// Created by sl on 8/2/17.
//

#define EIGEN_USE_THREADS
#define EIGEN_USE_GPU

#include "deform_conv3d.h"
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


REGISTER_OP("DeformConv3d")
        .Input("input: T")
        .Input("filter: T")
        .Input("offset: T")
        .Output("output: T")
        .Attr("strides: list(int)  = [1,1,1]")
        .Attr("dilatation_rates: list(int)  = [1,1,1]")
        .Attr("padding: {'SAME', 'VALID'} = 'VALID'")
        .Attr("T: {float, double}")
        .SetShapeFn([](InferenceContext *c) {
            //make sure the rank of input is right
            //NCLHW
            ShapeHandle input_shape;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 5, &input_shape));
            //N'CL'H'W'
            ShapeHandle filter_shape;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 5, &filter_shape));
            //NGL"H"W"L'H'W'3
            ShapeHandle offset_shape;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 9, &offset_shape));


            //calcute the output shape lhw
            vector<int64> output_size = {0, 0, 0};
            vector<int64> pads = {0, 0, 0};
            vector<int64> strides;
            Padding padding;
            TF_RETURN_IF_ERROR(c->GetAttr("strides", &strides));
            TF_RETURN_IF_ERROR(c->GetAttr("padding", &padding));
            for (int i = 0; i < 3; ++i) {
                TF_RETURN_IF_ERROR(GetWindowedOutputSize(
                        c->Value(c->Dim(input_shape, i + 2)), c->Value(c->Dim(filter_shape, i + 2)), strides[i],
                        padding, &output_size[i], &pads[i]));
            }
            int64 batch_size = c->Value(c->Dim(input_shape, 0));
            int64 filter_num = c->Value(c->Dim(filter_shape, 0));
            int64 filter_channel = c->Value(c->Dim(filter_shape, 1));
            int64 output_channel = filter_num * filter_channel;
            ShapeHandle output_shape = c->MakeShape(
                    {batch_size, output_channel, output_size[0], output_size[1], output_size[2]});

            //set output shape NC*N'L"H"W"
            c->set_output(0, output_shape);
            return Status::OK();
        });

// CPU specialization of actual computation.
template<typename T>
struct DeformConv3dFunctor<CPUDevice, T> {
    void operator()(const CPUDevice &d,
                    const T *data_im, const T *data_filter, const T *data_offset,
                    const vector<int64> &im_shape,
                    const vector<int64> &filter_shape,
                    const vector<int64> &offset_shape,
                    const vector<int64> &output_shape,
                    const vector<int64> &pad, const vector<int64> &stride, const vector<int64> &dilatation,
                    T *data_output) {
        cout << "using cpu.\n";
    };
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template<typename Device, typename T>
class DeformConv3dOp : public OpKernel {
public:
    explicit DeformConv3dOp(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("strides", &strides));
        OP_REQUIRES(context, strides.size() == 3,
                    errors::InvalidArgument("strides too large"));
        OP_REQUIRES_OK(context, context->GetAttr("dilatation_rates", &dilatation_rates));
        OP_REQUIRES(context, dilatation_rates.size() == 3,
                    errors::InvalidArgument("dilatation_rates too large"));
        OP_REQUIRES_OK(context, context->GetAttr("padding", &padding));

    }

    void Compute(OpKernelContext *context) override {
        // Grab the input tensor
        //NCLHW
        const Tensor &input = context->input(0);
        //C'L'H'W'
        const Tensor &filter = context->input(1);
        //NGL"H"W"L'H'W'3
        const Tensor &offset = context->input(2);

        //shape of output L"H"W"
        vector<int64> output_size = {0, 0, 0};
        vector<int64> pads = {0, 0, 0};
        int64 batch_size = input.dim_size(0);
        int64 input_channel = input.dim_size(1);
        vector<int64> input_size = {input.dim_size(2), input.dim_size(3), input.dim_size(4)};
        int64 filter_num = filter.dim_size(0);
        int64 filter_channel = filter.dim_size(1);
        vector<int64> filter_size = {filter.dim_size(2), filter.dim_size(3), filter.dim_size(4)};
        int64 offset_batch_size = offset.dim_size(0);
        int64 offset_group = offset.dim_size(1);
        vector<int64> offset_size = {offset.dim_size(2), offset.dim_size(3), offset.dim_size(4),
                                     offset.dim_size(5), offset.dim_size(6), offset.dim_size(7)};
        //check everything
        OP_REQUIRES(context, offset.dim_size(8) == 3,
                    errors::InvalidArgument("last dim_size of offset should be 3"));
        OP_REQUIRES(context, offset_batch_size == batch_size,
                    errors::InvalidArgument("batch size of offset"));
        OP_REQUIRES(context, input_channel == filter_channel,
                    errors::InvalidArgument("filter channel"));
        OP_REQUIRES(context, input_channel * filter_num % offset_group == 0, errors::InvalidArgument("offset group"));
        for (int i = 0; i < 3; ++i) {
            OP_REQUIRES_OK(context,
                           GetWindowedOutputSize(input_size[i], filter_size[i],
                                                 strides[i], padding, &(output_size[i]), &pads[i]));

            OP_REQUIRES(context, offset_size[i + 3] == filter_size[i],
                        errors::InvalidArgument("offset: ", offset_size[i+3]," vs filter", filter_size[i]));

            OP_REQUIRES(context, offset_size[i] == output_size[i],
                        errors::InvalidArgument("offset: ", offset_size[i]," vs output: ", output_size[i]));
        }

        const vector<int64> output_shape = {batch_size, filter_num * filter_channel,
                                            output_size[0], output_size[1], output_size[2]};
        const vector<int64> input_shape = {batch_size, input_channel, input_size[0], input_size[1], input_size[2]};
        const vector<int64> filter_shape = {filter_num, filter_channel, filter_size[0], filter_size[1], filter_size[2]};
        const vector<int64> offset_shape = {batch_size, offset_group, offset_size[0], offset_size[1], offset_size[2],
                                            offset_size[3], offset_size[4], offset_size[5], 3};

        // Create an output tensor
        Tensor *output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape(
                {output_shape[0], output_shape[1], output_shape[2], output_shape[3], output_shape[4]}), &output));
        T *output_ptr = output->template flat<T>().data();

        // Get the input and offset data ptr
        const T *input_ptr = input.template flat<T>().data();
        const T *offset_ptr = offset.template flat<T>().data();
        const T *filter_ptr = filter.template flat<T>().data();

        DeformConv3dFunctor<Device, T>()(
                context->eigen_device<Device>(),
                input_ptr, filter_ptr, offset_ptr,
                input_shape, filter_shape, offset_shape, output_shape,
                pads, strides, dilatation_rates,
                output_ptr
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
      Name("DeformConv3d")                          \
      .Device(DEVICE_GPU)                                        \
      .TypeConstraint<T>("T"),                                   \
      DeformConv3dOp<GPUDevice, T>);
REGISTER_GPU(float);
//REGISTER_GPU(int64);
REGISTER_GPU(double);
#endif  // GOOGLE_CUDA

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("DeformConv3d")                          \
      .Device(DEVICE_CPU)                                        \
      .TypeConstraint<T>("T"),                                   \
      DeformConv3d<CPUDevice, T>);
//REGISTER_CPU(float);
//REGISTER_CPU ;
//REGISTER_CPU(double);