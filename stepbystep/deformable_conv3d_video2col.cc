//
// Created by sl on 8/2/17.
//

#define EIGEN_USE_THREADS

#include "deformable_conv3d_video2col.h"
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


REGISTER_OP("DeformableConv3dVideo2col")
        .Input("input: T")
        .Input("filter: T")
        .Input("offset: T")
        .Output("output: T")
        .Attr("strides: list(int) = [1,1,1]")
        .Attr("dilations: list(int) = [1,1,1]")
        .Attr("deformable_groups: int = 1")
        .Attr("padding: {'SAME', 'VALID'} = 'VALID'")
        .Attr("T: {float, double, int32} = DT_INT32")
        .SetShapeFn([](InferenceContext *c) {
//            c->set_output(0, c->input(0));
            ShapeHandle input_shape;//nclhw
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 5, &input_shape));
            ShapeHandle filter_shape;//clhw
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &filter_shape));//withrank 确定维数正确,5维
            ShapeHandle offset_shape;//glhwd
            TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 5, &offset_shape));

            vector<int> strides, dilations;
            int deformable_groups;
            Padding padding;
            TF_RETURN_IF_ERROR(c->GetAttr("strides", &strides));
            TF_RETURN_IF_ERROR(c->GetAttr("dilations", &dilations));
            TF_RETURN_IF_ERROR(c->GetAttr("deformable_groups", &deformable_groups));
            TF_RETURN_IF_ERROR(c->GetAttr("padding", &padding));

            vector<int64> output_shape(3);
            vector<int64> padding_after(3);
            for (int i = 0; i < 3; ++i) {
                TF_RETURN_IF_ERROR(GetWindowedOutputSize(
                        c->Value(c->Dim(input_shape, i + 2)), c->Value(c->Dim(filter_shape, i + 1)), strides[i],
                        padding, &output_shape[i], &padding_after[i]));
            }

            cout << "output shape: ";
            std::copy(begin(output_shape), end(output_shape), std::ostream_iterator<int64>(std::cout, " "));
//            cout << "offset shape: ";
//            std::copy(begin(offset_shape), end(offset_shape), std::ostream_iterator<int64>(std::cout, " "));
//            cout << "filter shape: ";
//            std::copy(begin(filter_shape), end(filter_shape), std::ostream_iterator<int64>(std::cout, " "));

            //TODO
            int64 batch_size_dim = c->Value(c->Dim(input_shape, 0));
            int64 out_channels = c->Value(c->Dim(input_shape, 1));
//            int64 output_lens = c->Value(c->Dim(input_shape, 2));
//            int64 output_cols = c->Value(c->Dim(input_shape, 3));
//            int64 output_rows = c->Value(c->Dim(input_shape, 4));
//
            ShapeHandle output_shapes = c->MakeShape(
                    {batch_size_dim, out_channels, output_shape[0], output_shape[1], output_shape[2],
                     c->Value(c->Dim(filter_shape, 1)) *
                     c->Value(c->Dim(filter_shape, 2)) *
                     c->Value(c->Dim(filter_shape, 3))});

            cout << "col shape: ";
            cout << output_shape[0] << output_shape[1] << output_shape[2] <<
                 c->Value(c->Dim(filter_shape, 1)) *
                 c->Value(c->Dim(filter_shape, 2)) *
                 c->Value(c->Dim(filter_shape, 3)) << endl;
            c->set_output(0, output_shapes);
            return Status::OK();
        });

// CPU specialization of actual computation.
template<typename T>
struct DeformableConv3dVideo2colFunctor<CPUDevice, T> {
    void operator()(const CPUDevice &d,
                    const T *&data_im, const T *&data_offset,
                    const TensorShape &im_shape, const TensorShape &col_shape, const TensorShape &kernel_shape,
                    const vector<int> &pad, const vector<int> &stride, const vector<int> &dilation,
                    int deformable_group, T *&data_col) {
        cout << "cpu";
    };
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template<typename Device, typename T>
class DeformableConv3dVideo2colOp : public OpKernel {
public:
    explicit DeformableConv3dVideo2colOp(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("strides", &strides));
        OP_REQUIRES(context, strides.size() == 3,
                    errors::InvalidArgument("strides too large"));
        OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilations));
        OP_REQUIRES(context, dilations.size() == 3,
                    errors::InvalidArgument("dilations too large"));
        OP_REQUIRES_OK(context, context->GetAttr("deformable_groups", &deformable_groups));
        OP_REQUIRES_OK(context, context->GetAttr("padding", &padding));

        cout << '\n' << "dilations:";
        std::copy(begin(dilations), end(dilations), std::ostream_iterator<T>(std::cout, " "));
        cout << '\n' << "strides:";
        std::copy(begin(strides), end(strides), std::ostream_iterator<T>(std::cout, " "));
        cout << "padding: " << padding << '\n' << "groups: " << deformable_groups << endl;

    }

    void Compute(OpKernelContext *context) override {
        // Grab the input tensor
        //nclhw
        const Tensor &input = context->input(0);
        const TensorShape &input_shape = input.shape();
        const TensorShape &input_shape_one_batch = TensorShape({
                                                                       input_shape.dim_size(1), input_shape.dim_size(2),
                                                                       input_shape.dim_size(3), input_shape.dim_size(4),
                                                               });
        //clhw
        const Tensor &filter = context->input(1);
        const TensorShape &filter_shape = filter.shape();
        //glhwd
        const Tensor &offset = context->input(2);
        const TensorShape &offset_shape = offset.shape();

        int64 output_shape[3];
        int64 padding_after[3];
        int batch_size = input.dim_size(0);
        int input_channel = input.dim_size(1);

        //TODO
        for (int i = 0; i < 3; ++i) {
            OP_REQUIRES_OK(context,
                           GetWindowedOutputSize(input.dim_size(i + 2), filter.dim_size(i + 1),
                                                 strides[i], padding, &output_shape[i], &padding_after[i])
            );
        }

        //col buffer
        const TensorShape col_shape = TensorShape(
                {batch_size, input_channel,
                 output_shape[0], output_shape[1], output_shape[2],
                 ProdShape(filter_shape, 1, filter_shape.dims())});
        const TensorShape col_shape_one_batch = TensorShape(
                {input_channel,
                 output_shape[0], output_shape[1], output_shape[2],
                 ProdShape(filter_shape, 1, filter_shape.dims())});
        Tensor col;
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, col_shape, &col));

        // Create an output tensor  col tensor
        //TODO: 暂时取值为col
        Tensor *output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, col_shape, &output));

        //TODO:为什么一定要用int64
        vector<int> pad = {padding_after[0], padding_after[1], padding_after[2]};
        // Do the computation.
        T *col_base_ptr = col.template flat<T>().data();
        const T *input_base_ptr = input.template flat<T>().data();

        const T *offset_ptr = offset.template flat<T>().data();

        cout << ".cc file            pad: " << pad[0]
             << "  strides: " << strides[0]
             << col.NumElements() << endl;
//        for (int j = 0; j < input.NumElements(); ++j) {
//            cout << j << " " << input.template vec<T>()(j);
//        }
//        auto a = col.template flat<T>();
//        cout<<a<<'\n';
//        for (int j = 0; j < a.size(); ++j) {
//            cout << a <<"   ";
//        }
//        cout<<'\n';
        for (int i = 0; i < batch_size; ++i) {
//            int groups = n*input_channel/deformable_groups;
            const T *input_ptr = input_base_ptr + i * ProdShape(input_shape, 1, input_shape.dims());

            T *col_ptr = col_base_ptr + i * ProdShape(col_shape, 1, col_shape.dims());
            DeformableConv3dVideo2colFunctor<Device, T>()(
                    context->eigen_device<Device>(),
                    input_ptr,
                    offset_ptr,
                    input_shape_one_batch, col_shape_one_batch, filter_shape,
                    pad, strides, dilations, deformable_groups,
                    col_ptr
            );

        }
        output = &col;
    }

private:
    vector<int> strides;
    vector<int> dilations;
    int deformable_groups;
    Padding padding;
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("DeformableConv3dVideo2col").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      DeformableConv3dVideo2colOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(int32);
REGISTER_CPU(double);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("DeformableConv3dVideo2col")                          \
      .Device(DEVICE_GPU)                                        \
      .TypeConstraint<T>("T"),                                   \
      DeformableConv3dVideo2colOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(int32);
REGISTER_GPU(double);
#endif  // GOOGLE_CUDA