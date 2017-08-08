//
// Created by sl on 8/2/17.
//

#define EIGEN_USE_THREADS
#define EIGEN_USE_GPU

#include "deformable_conv3d.h"
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


REGISTER_OP("DeformableConv3d")
        .Input("input: T")
        .Input("filter: T")
        .Input("offset: T")
        .Output("output: T")
        .Attr("strides: list(int) = [1,1,1]")
        .Attr("dilatation_rates: list(int) = [1,1,1]")
        .Attr("deformable_groups: int = 1")
        .Attr("padding: {'SAME', 'VALID'} = 'VALID'")
        .Attr("T: {float, double, int64}")
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
            //get the attributes
            vector<int64> strides, dilatation_rates;
            int64 deformable_groups;
            Padding padding;
            TF_RETURN_IF_ERROR(c->GetAttr("strides", &strides));
            TF_RETURN_IF_ERROR(c->GetAttr("dilatation_rates", &dilatation_rates));
            TF_RETURN_IF_ERROR(c->GetAttr("deformable_groups", &deformable_groups));
            TF_RETURN_IF_ERROR(c->GetAttr("padding", &padding));

            //calcute the output shape lhw
            vector<int64> output_shape = {0, 0, 0};
            vector<int64> pads = {0, 0, 0};
            for (int i = 0; i < 3; ++i) {
                TF_RETURN_IF_ERROR(GetWindowedOutputSize(
                        c->Value(c->Dim(input_shape, i + 2)), c->Value(c->Dim(filter_shape, i + 1)), strides[i],
                        padding, &output_shape[i], &pads[i]));
            }

            int64 batch_size = c->Value(c->Dim(input_shape, 0));
            int64 col_channels = c->Value(c->Dim(input_shape, 1));
            int64 output_channels = col_channels * c->Value(c->Dim(filter_shape, 0));
            int64 volume_filter = c->Value(c->Dim(filter_shape, 1)) *
                                  c->Value(c->Dim(filter_shape, 2)) * c->Value(c->Dim(filter_shape, 3));
            //test attr deformable groups
            if (output_channels % deformable_groups != 0) {
                return errors::InvalidArgument("deformable_groups should be divided by output_channels");
            }
            //check the single value of input lhw
            if (c->Value(c->Dim(input_shape, 2)) % 2 != 1 || c->Value(c->Dim(input_shape, 3)) % 2 != 1
                || c->Value(c->Dim(input_shape, 4)) % 2 != 1) {
                return errors::InvalidArgument("the input is not singular");
            }
            //check the single value of filter lhw
            if (c->Value(c->Dim(filter_shape, 1)) % 2 != 1 || c->Value(c->Dim(filter_shape, 2)) % 2 != 1
                || c->Value(c->Dim(filter_shape, 3)) % 2 != 1) {
                return errors::InvalidArgument("the input is not singular");
            }

            //check the offset value of lhw
            if (c->Value(c->Dim(input_shape, 2)) != c->Value(c->Dim(offset_shape, 1))
                || c->Value(c->Dim(input_shape, 3)) != c->Value(c->Dim(offset_shape, 2))
                || c->Value(c->Dim(input_shape, 4)) != c->Value(c->Dim(offset_shape, 3))) {
                return errors::InvalidArgument("the offset is not same as input");
            }
            //test offset depth
            if (c->Value(c->Dim(offset_shape, 4)) != volume_filter) {
                return errors::InvalidArgument("the depth of offset is not right");
            }
            //test offset last dim
            if (c->Value(c->Dim(offset_shape, 5)) != 3) {
                return errors::InvalidArgument("the last dim of offset is not right");
            }

            //the output shape of col
            ShapeHandle col_shapes = c->MakeShape(
                    {batch_size, col_channels, output_shape[0], output_shape[1], output_shape[2], volume_filter});

            ShapeHandle output_shapes = c->MakeShape(
                    {batch_size, output_channels, output_shape[0], output_shape[1], output_shape[2]});

            //set output shape
            c->set_output(0, output_shapes);
            return Status::OK();
        });

// CPU specialization of actual computation.
template<typename T>
struct DeformableConv3dFunctor<CPUDevice, T> {
    void operator()(const CPUDevice &d,
                    const T *data_im, const T *data_offset,
                    const TensorShape &im_shape, const TensorShape &col_shape, const TensorShape &kernel_shape,
                    const vector<int64> &pad, const vector<int64> &stride, const vector<int64> &dilation,
                    int64 deformable_group, T *data_col, T *data_output, const T *data_kernel) {
        cout << "using cpu.\n";
    };
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template<typename Device, typename T>
class DeformableConv3dOp : public OpKernel {
public:
    explicit DeformableConv3dOp(OpKernelConstruction *context) : OpKernel(context) {
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
        // Grab the input tensor
        //nclhw
        const Tensor &input = context->input(0);
        const TensorShape &input_shape = input.shape();
        //clhw
        const Tensor &filter = context->input(1);
        const TensorShape &filter_shape = filter.shape();
        //glhwd3
        const Tensor &offset = context->input(2);
        const TensorShape &offset_shape = offset.shape();

        //lhw of output
        vector<int64> output_shape = {0, 0, 0};
        vector<int64> pads = {0, 0, 0};
        int64 batch_size = input.dim_size(0);
        int64 input_channel = input.dim_size(1);
        int64 filter_channel = filter.dim_size(0);

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

        //output shape
        const TensorShape output_shapes = TensorShape(
                {batch_size, input_channel * filter_channel,
                 output_shape[0], output_shape[1], output_shape[2]});

        // Create an output tensor
        Tensor *output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shapes, &output));
        T *output_ptr = output->template flat<T>().data();

        // Create a temp for col.
        Tensor col;
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, col_shape, &col));
        T *col_base_ptr = col.template flat<T>().data();

        // Get the input and offset data ptr
        const T *input_ptr = input.template flat<T>().data();
        const T *offset_ptr = offset.template flat<T>().data();
        const T *filter_ptr = filter.template flat<T>().data();

        DeformableConv3dFunctor<Device, T>()(
                context->eigen_device<Device>(),
                input_ptr,
                offset_ptr,
                input_shape, col_shape, filter_shape,
                pads, strides, dilatation_rates, deformable_groups,
                col_base_ptr,
                output_ptr,
                filter_ptr
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
      Name("DeformableConv3d")                          \
      .Device(DEVICE_GPU)                                        \
      .TypeConstraint<T>("T"),                                   \
      DeformableConv3dOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(int64);
REGISTER_GPU(double);
#endif  // GOOGLE_CUDA

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("DeformableConv3d")                          \
      .Device(DEVICE_CPU)                                        \
      .TypeConstraint<T>("T"),                                   \
      DeformableConv3d<CPUDevice, T>);
//REGISTER_CPU(float);
//REGISTER_CPU(int64);
//REGISTER_CPU(double);