//#define EIGEN_USE_THREADS
#include <vector>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/platform/stream_executor.h"

#include "deform_conv.h"
#include "deform_conv_util.h"


using namespace tensorflow;
using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
//------------------------------------------Register-----------------------------
namespace {
    REGISTER_OP("DeformConvOp")
            .Input("x: T") //T是类型
            .Input("filter: T")
            .Input("offset: T")
            .Output("output: T")
            .Attr("T: {half, float, double}")
            .Attr("strides: list(int)")
            .Attr("rates: list(int)")
            .Attr("num_groups: int")
            .Attr("deformable_group: int")
            .Attr(GetPaddingAttrString())
            .Attr("data_format: { 'NHWC', 'NCHW' } = 'NCHW' ")
            .SetShapeFn([](InferenceContext *c) {
                ShapeHandle input_shape;
                //TODO:shilei
                TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_shape));
                ShapeHandle filter_shape;
                TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &filter_shape));
                ShapeHandle offset_shape;
                TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 4, &offset_shape));

                std::vector<int32> strides;
                TF_RETURN_IF_ERROR(c->GetAttr("strides", &strides));
                if (strides.size() != 4) {
                    return errors::InvalidArgument(
                            "Deformconv requires the stride attribute to contain 4 values, but "
                                    "got: ",
                            strides.size());
                }

                std::vector<int32> rates;
                TF_RETURN_IF_ERROR(c->GetAttr("rates", &rates));
                if (rates.size() != 4) {
                    return errors::InvalidArgument(
                            "Deformconv requires the rates attribute to contain 4 values, but "
                                    "got: ",
                            rates.size());
                }
                string data_format;
                TensorFormat data_format_;
                TF_RETURN_IF_ERROR(c->GetAttr("data_format", &data_format));
                FormatFromString(data_format, &data_format_);
                const int32 stride_rows = GetTensorDim(strides, data_format_, 'H');
                const int32 stride_cols = GetTensorDim(strides, data_format_, 'W');

                const int32 rate_rows = GetTensorDim(rates, data_format_, 'H');
                const int32 rate_cols = GetTensorDim(rates, data_format_, 'W');


                int groups;
                TF_RETURN_IF_ERROR(c->GetAttr("num_groups", &groups));
                int deform_groups;
                TF_RETURN_IF_ERROR(c->GetAttr("deformable_group", &deform_groups));

                DimensionHandle batch_size_dim = c->Dim(input_shape, 0);
                DimensionHandle in_depths_dim = c->Dim(input_shape, 1);
                DimensionHandle in_rows_dim = c->Dim(input_shape, 2);
                DimensionHandle in_cols_dim = c->Dim(input_shape, 3);
                DimensionHandle filter_rows_dim = c->Dim(filter_shape, 2);
                DimensionHandle filter_cols_dim = c->Dim(filter_shape, 3);
                DimensionHandle filter_depth_dim = c->Dim(filter_shape, 1);
                DimensionHandle output_depth_dim = c->Dim(filter_shape, 0);
                DimensionHandle multiplied_depth;
                DimensionHandle depth_per_dfgps;
                auto filter_row = c->Value(filter_rows_dim);
                auto filter_col = c->Value(filter_cols_dim);
                auto offset_dpt = c->Value(c->Dim(offset_shape, 1));
                if ((offset_dpt % (filter_row * filter_col) != 0) ||
                    (offset_dpt / (2 * filter_row * filter_col) != deform_groups)) {
                    return errors::InvalidArgument(
                            "Deformconv requires the offset compatible with filter, but "
                                    "got: ",
                            c->DebugString(offset_shape));
                }
                TF_RETURN_IF_ERROR(
                        c->Multiply(filter_depth_dim, groups, &multiplied_depth));
                TF_RETURN_IF_ERROR(c->Divide(filter_depth_dim, deform_groups, true, &depth_per_dfgps));
                TF_RETURN_IF_ERROR(c->Divide(in_depths_dim, deform_groups, true, &depth_per_dfgps));


                if (!c->ValueKnown(in_rows_dim) || !c->ValueKnown(in_cols_dim) ||
                    !c->ValueKnown(filter_rows_dim) || !c->ValueKnown(filter_cols_dim)) {
                    ShapeHandle output_shape =
                            c->MakeShape({batch_size_dim, output_depth_dim, InferenceContext::kUnknownDim,
                                          InferenceContext::kUnknownDim
                                         });
                    c->set_output(0, output_shape);
                    return Status::OK();
                }
                DimensionHandle unused;
                TF_RETURN_IF_ERROR(
                        c->Merge(c->Dim(input_shape, 1), multiplied_depth, &unused));

                auto in_rows = c->Value(in_rows_dim);
                auto in_cols = c->Value(in_cols_dim);
                auto filter_rows = c->Value(filter_rows_dim);
                auto filter_cols = c->Value(filter_cols_dim);
                auto filter_rows_eff = filter_rows + (filter_rows - 1) * (rate_rows - 1);
                auto filter_cols_eff = filter_cols + (filter_cols - 1) * (rate_cols - 1);

                Padding padding;
                TF_RETURN_IF_ERROR(c->GetAttr("padding", &padding));

                int64 output_rows, output_cols;
                int64 padding_before, padding_after;
                TF_RETURN_IF_ERROR(GetWindowedOutputSizeVerbose(
                        in_rows, filter_rows_eff, stride_rows, padding, &output_rows,
                        &padding_before, &padding_after));
                TF_RETURN_IF_ERROR(GetWindowedOutputSizeVerbose(
                        in_cols, filter_cols_eff, stride_cols, padding, &output_cols,
                        &padding_before, &padding_after));

                ShapeHandle output_shape = c->MakeShape(
                        {batch_size_dim, output_depth_dim, output_rows, output_cols});
                c->set_output(0, output_shape);
                return Status::OK();
            })
            .Doc(R"doc(
only support NCHW now
)doc");
}

//-----------------------------------------CPUDevice kernel implement----------------------------------
template<typename DType>
struct deformable_im2col<CPUDevice, DType> {
    void operator()(const Device &d,
                    const DType *data_im, const DType *data_offset,
                    const TensorShape &im_shape, const TensorShape &col_shape, const TensorShape &kernel_shape,
                    const TensorShape &pad, const TensorShape &stride, const TensorShape &dilation,
                    const int deformable_group, DType *data_col) {};
};
void cpumatmul(){};
//-------------------------------------------------------计算-----------------------------------------------
namespace {
    template<typename T>
    perftools::gputools::DeviceMemory<T> AsDeviceMemory(const T *cuda_memory) {
        perftools::gputools::DeviceMemoryBase wrapped(const_cast<T *>(cuda_memory));
        perftools::gputools::DeviceMemory<T> typed(wrapped);
        return typed;
    }
}
namespace functor{
class CublasScratchAllocator : public perftools::gputools::ScratchAllocator {
public:
    using Stream = ::perftools::gputools::Stream;
    using DeviceMemoryBytes = ::perftools::gputools::DeviceMemory<uint8>;

    CublasScratchAllocator(OpKernelContext *context) : context_(context) {}

    int64 GetMemoryLimitInBytes(Stream *stream) override { return -1; }

    perftools::gputools::port::StatusOr<DeviceMemoryBytes> AllocateBytes(
            Stream *stream, int64 byte_size) override {
        Tensor temporary_memory;

        Status allocation_status(context_->allocate_temp(
                DT_UINT8, TensorShape({byte_size}), &temporary_memory));
        if (!allocation_status.ok()) {
            return perftools::gputools::port::StatusOr<DeviceMemoryBytes>(
                    DeviceMemoryBytes::MakeFromByteSize(nullptr, 0));
        }
        // Hold the reference of the allocated tensors until the end of the
        // allocator.
        allocated_tensors_.push_back(temporary_memory);
        return perftools::gputools::port::StatusOr<DeviceMemoryBytes>(
                DeviceMemoryBytes::MakeFromByteSize(
                        temporary_memory.flat<uint8>().data(),
                        temporary_memory.flat<uint8>().size()));
    }

private:
    OpKernelContext *context_;
    std::vector<Tensor> allocated_tensors_;
};


template<typename DType>
struct LaunchBatchMatMul {
    static void Launch(OpKernelContext *context, const TensorShape &in_x_shape,
                           const TensorShape &in_y_shape,
                           const DType *in_x_ptr,
                           const DType *in_y_ptr, bool adj_x, bool adj_y, DType *out) {
        constexpr perftools::gputools::blas::Transpose kTranspose =
                is_complex<DType>::value
                ? perftools::gputools::blas::Transpose::kConjugateTranspose
                : perftools::gputools::blas::Transpose::kTranspose;
        perftools::gputools::blas::Transpose trans[] = {
                perftools::gputools::blas::Transpose::kNoTranspose, kTranspose};
        const uint64 m = in_x_shape.dim_size(adj_x ? 2 : 1);
        const uint64 k = in_x_shape.dim_size(adj_x ? 1 : 2);
        const uint64 n = in_y_shape.dim_size(adj_y ? 1 : 2);
        const uint64 batch_size = in_x_shape.dim_size(0);
        auto blas_transpose_a = trans[adj_x];
        auto blas_transpose_b = trans[adj_y];

        auto *stream = context->op_device_context()->stream();
        OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

        typedef perftools::gputools::DeviceMemory<DType> DeviceMemoryType;
        std::vector<DeviceMemoryType> a_device_memory;
        std::vector<DeviceMemoryType> b_device_memory;
        std::vector<DeviceMemoryType> c_device_memory;
        std::vector<DeviceMemoryType *> a_ptrs;
        std::vector<DeviceMemoryType *> b_ptrs;
        std::vector<DeviceMemoryType *> c_ptrs;
        a_device_memory.reserve(batch_size);
        b_device_memory.reserve(batch_size);
        c_device_memory.reserve(batch_size);
        a_ptrs.reserve(batch_size);
        b_ptrs.reserve(batch_size);
        c_ptrs.reserve(batch_size);
        auto *a_base_ptr = in_x_ptr;
        auto *b_base_ptr = in_y_ptr;
        // auto* c_base_ptr = out->template flat<Scalar>().data();
        auto *c_base_ptr = out;
        for (int64 i = 0; i < batch_size; ++i) {
            a_device_memory.push_back(AsDeviceMemory(a_base_ptr + i * m * k));
            b_device_memory.push_back(AsDeviceMemory(b_base_ptr + i * k * n));
            c_device_memory.push_back(AsDeviceMemory(c_base_ptr + i * m * n));
            a_ptrs.push_back(&a_device_memory.back());
            b_ptrs.push_back(&b_device_memory.back());
            c_ptrs.push_back(&c_device_memory.back());
        }

        // Cublas does
        // C = A x B
        // where A, B and C are assumed to be in column major.
        // We want the output to be in row-major, so we can compute
        // C'= B' x A', where' stands for transpose (not adjoint).
        // TODO(yangzihao): Choose the best of the three strategies using autotune.
        if (batch_size == 1) {
            // This is a regular matrix*matrix or matrix*vector multiply. Avoid the
            // overhead of the scratch allocator and the batch interface.
            if (n == 1 &&
                blas_transpose_b !=
                perftools::gputools::blas::Transpose::kConjugateTranspose &&
                blas_transpose_a !=
                perftools::gputools::blas::Transpose::kConjugateTranspose) {
                // This is a matrix*vector multiply so use GEMV to compute A * b.
                // Here we are multiplying in the natural order, so we have to flip
                // the transposition flag to compensate for the tensor being stored
                // row-major. Since GEMV doesn't provide a way to just conjugate an
                // argument, we have to defer those cases to GEMM below.
                //TODO：真正的计算
                auto gemv_trans_a =
                        blas_transpose_a == perftools::gputools::blas::Transpose::kTranspose
                        ? perftools::gputools::blas::Transpose::kNoTranspose
                        : perftools::gputools::blas::Transpose::kTranspose;
                bool blas_launch_status =
                        stream->ThenBlasGemv(gemv_trans_a, adj_x ? m : k, adj_x ? k : m,
                                             static_cast<DType>(1.0), *(a_ptrs[0]),
                                             adj_x ? m : k, *(b_ptrs[0]), 1,
                                             static_cast<DType>(0.0), c_ptrs[0], 1).ok();
                if (!blas_launch_status) {
                    context->SetStatus(errors::Internal(
                            "Blas xGEMV launch failed : a.shape=", in_x_shape.DebugString(),
                            ", b.shape=", in_y_shape.DebugString(), ", m=", m, ", n=", n,
                            ", k=", k));
                }
            } else {
                bool blas_launch_status =
                        stream->ThenBlasGemm(blas_transpose_b, blas_transpose_a, n, m, k,
                                             static_cast<DType>(1.0), *(b_ptrs[0]),
                                             adj_y ? k : n, *(a_ptrs[0]), adj_x ? m : k,
                                             static_cast<DType>(0.0), c_ptrs[0], n).ok();
                if (!blas_launch_status) {
                    context->SetStatus(errors::Internal(
                            "Blas xGEMM launch failed : a.shape=", in_x_shape.DebugString(),
                            ", b.shape=", in_y_shape.DebugString(), ", m=", m, ", n=", n,
                            ", k=", k));
                }
            }
        } else {
            CublasScratchAllocator scratch_allocator(context);
            bool blas_launch_status =
                    stream->ThenBlasGemmBatchedWithScratch(
                            blas_transpose_b, blas_transpose_a, n, m, k,
                            static_cast<DType>(1.0), b_ptrs, adj_y ? k : n, a_ptrs,
                            adj_x ? m : k, static_cast<DType>(0.0), c_ptrs, n,
                            batch_size, &scratch_allocator).ok();
            if (!blas_launch_status) {
                context->SetStatus(errors::Internal(
                        "Blas xGEMMBatched launch failed : a.shape=",
                        in_x_shape.DebugString(),
                        ", b.shape=", in_y_shape.DebugString(), ", m=", m, ", n=", n,
                        ", k=", k, ", batch_size=", batch_size));
            }
        }
    }
};
}
//-------------------------------------OpKernel----------------------------------------------------
template<typename Device, typename T>
class DeformConvOp : public OpKernel {
public:
    explicit DeformConvOp(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
        OP_REQUIRES_OK(context, context->GetAttr("rates", &rates_));
        string data_format;
        OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
        // TensorFormat data_format_;
        OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                    errors::InvalidArgument("Invalid data format"));
        OP_REQUIRES(context, strides_.size() == 4,
                    errors::InvalidArgument("Sliding window strides field must "
                                                    "specify 4 dimensions"));
        const int64 stride_n = GetTensorDim(strides_, data_format_, 'N');
        const int64 stride_c = GetTensorDim(strides_, data_format_, 'C');
        OP_REQUIRES(
                context, stride_n == 1 && stride_c == 1,
                errors::InvalidArgument("Current implementation does not yet support "
                                                "strides in the batch and depth dimensions."));
        OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
        const int64 stride_H = GetTensorDim(strides_, data_format_, 'H');
        const int64 stride_W = GetTensorDim(strides_, data_format_, 'W');
        OP_REQUIRES_OK(context, context->GetAttr("num_groups", &num_groups));
        OP_REQUIRES_OK(context, context->GetAttr("deformable_group", &deformable_group));

    }

    void Compute(OpKernelContext *context) override {
        CHECK(data_format_ == FORMAT_NCHW) << "Generic conv implementation only "
                    "supports NCHW tensor format for now.";
        // Input tensor is of the following dimensions:
        // [ batch, in_rows, in_cols, in_depth ]

        const Tensor &input = context->input(0);
        const TensorShape &ishape = input.shape();
        // Input filter is of the following dimensions:
        // [ out_depth, in_depth, filter_rows, filter_cols]
        const Tensor &filter = context->input(1);

        const Tensor &offset = context->input(2);
        const TensorShape &offset_shape = offset.shape();
        int num_filter = filter.dim_size(0);
        // param_->num_filter = depth_out;
        // For 2D convolution, there should be 4 dimensions. TODO：3d-5
        OP_REQUIRES(context, input.dims() == 4,
                    errors::InvalidArgument("input must be 4-dimensional",
                                            input.shape().DebugString()));
        OP_REQUIRES(context, filter.dims() == 4,
                    errors::InvalidArgument("filter must be 4-dimensional: ",
                                            filter.shape().DebugString()));
        OP_REQUIRES(context, offset.dims() == 4,
                    errors::InvalidArgument("offset must be 4-dimensional: ",
                                            filter.shape().DebugString()));
        for (int i = 0; i < 3; i++) {
            OP_REQUIRES(context, FastBoundsCheck(filter.dim_size(i),//检查滤波器是否超过数字边界
                                                 std::numeric_limits<int>::max()),
                        errors::InvalidArgument("filter too large"));
        }

        // The last dimension for input is in_depth. It must be the same as the
        // filter's in_depth.
        const int64 in_depth = GetTensorDim(input, data_format_, 'C');
        OP_REQUIRES(
                context, in_depth == filter.dim_size(1) * num_groups,
                errors::InvalidArgument("input and filter must have the same depth: ",
                                        in_depth, " vs ", filter.dim_size(1)));
        OP_REQUIRES(
                context, offset_shape.dim_size(1) % (filter.dim_size(2) * filter.dim_size(3)) == 0,
                errors::InvalidArgument("offset channels must divide deformable group size: ",
                                        offset_shape.dim_size(1), " vs ", filter.dim_size(2) * filter.dim_size(3)));
        OP_REQUIRES(
                context, num_filter % num_groups == 0,
                errors::InvalidArgument("num_filter must divide deformable group size: ",
                                        filter.dim_size(0), " vs ", num_groups));

        // The second dimension for input is rows/height.
        // The first dimension for filter is rows/height.
        const int64 input_rows_raw = GetTensorDim(input, data_format_, 'H');
        OP_REQUIRES(context, FastBoundsCheck(input_rows_raw,
                                             std::numeric_limits<int>::max()),
                    errors::InvalidArgument("Input rows too large"));
        const int input_rows = static_cast<int>(input_rows_raw);
        const int filter_rows = static_cast<int>(filter.dim_size(2));

        // The third dimension for input is columns/width.
        // The second dimension for filter is columns/width.
        const int64 input_cols_raw = GetTensorDim(input, data_format_, 'W');
        OP_REQUIRES(context, FastBoundsCheck(input_cols_raw,
                                             std::numeric_limits<int>::max()),
                    errors::InvalidArgument("Input cols too large"));
        const int input_cols = static_cast<int>(input_cols_raw);
        const int filter_cols = static_cast<int>(filter.dim_size(3));

        // The first dimension for input is batch.
        const int64 batch_raw = GetTensorDim(input, data_format_, 'N');
        OP_REQUIRES(context,
                    FastBoundsCheck(batch_raw, std::numeric_limits<int>::max()),
                    errors::InvalidArgument("batch is too large"));
        const int batch = static_cast<int>(batch_raw);

        // For now we take the stride from the second and third dimensions only (we
        // do not support striding on the batch or depth dimension).
        const int stride_rows = GetTensorDim(strides_, data_format_, 'H');
        const int stride_cols = GetTensorDim(strides_, data_format_, 'W');
        const int rate_rows = GetTensorDim(rates_, data_format_, 'H');
        const int rate_cols = GetTensorDim(rates_, data_format_, 'W');

        int64 out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;
        //TODO:计算padding厚的大小
        OP_REQUIRES_OK(context,
                       GetWindowedOutputSize(input_rows, filter_rows, stride_rows,
                                             padding_, &out_rows, &pad_rows));
        OP_REQUIRES_OK(context,
                       GetWindowedOutputSize(input_cols, filter_cols, stride_cols,
                                             padding_, &out_cols, &pad_cols));
        TensorShape pad({static_cast<int>(pad_rows), static_cast<int>(pad_cols)});
        TensorShape stride({stride_rows, stride_cols});
        TensorShape kernels({filter_rows, filter_cols});
        TensorShape rates({rate_rows, rate_cols});
        TensorShape out_shape = ShapeFromFormat(data_format_, batch, out_rows, out_cols, num_filter);
        //结构体
        auto temp = DeformConvParam(kernels, stride, pad, rates, num_groups, num_filter, true);
        this->param_ = &temp;
        // LOG(INFO)<<"rates "<<(this->param_->rates)[0]<<" "<<(this->param_->rates)[1];
        //不知道在做什么
        LayerSetUp(ishape, offset_shape, out_shape);

        int M = conv_out_channels_ / group_;
        int N = conv_out_spatial_dim_;
        int K = kernel_dim_;
        Tensor weight_3d;  //卷积核展开
        OP_REQUIRES(context,
                    weight_3d.CopyFrom(filter, TensorShape({group_, M, K})),
                    errors::InvalidArgument("shape doesn't match"));
        const T *weight_3d_ptr = weight_3d.template flat<T>().data();
        Tensor *output_4d = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output_4d));
        T *output_4d_ptr = output_4d->template flat<T>().data();
        // this two shape size are equal
        auto col_buf_3d_shape = TensorShape({group_, K, N});
        auto col_buf_shape = TensorShape(
                {conv_in_channels_ * param_->kernel.dim_size(0) * param_->kernel.dim_size(1), out_rows, out_cols});
        Tensor col_buffer_3d;
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, col_buf_3d_shape, &col_buffer_3d));
        auto in_data_ptr = input.template flat<T>().data();
        auto offset_ptr = offset.template flat<T>().data();
        auto col_buffer_3d_ptr = col_buffer_3d.template flat<T>().data();
        const Device &d = context->eigen_device<Device>();

        for (int n = 0; n < num_; ++n) {
            // transform image to col_buffer_3d in order to use gemm 将图片deformable并展开
            deformable_im2col<Device, T>()(d, in_data_ptr + n * input_dim_,
                                           offset_ptr + n * input_offset_dim_, ishape,
                                           col_buf_shape, (this->param_->kernel),
                                           (this->param_->pad), (this->param_->stride),
                                           (this->param_->rates), deformable_group,
                                           col_buffer_3d_ptr);
            // Tensor output_3d = output_4d->Slice(n, n+1);
            T *output_3d_ptr = output_4d_ptr + n * output_dim_;
            functor::LaunchBatchMatMul<T>::Launch(context, weight_3d.shape(), col_buffer_3d.shape(), weight_3d_ptr,
                                         col_buffer_3d_ptr, false, false, output_3d_ptr);
        }


        VLOG(2) << "Conv2D: in_depth = " << in_depth
                << ", input_cols = " << input_cols
                << ", filter_cols = " << filter_cols
                << ", input_rows = " << input_rows
                << ", filter_rows = " << filter_rows
                << ", stride_rows = " << stride_rows
                << ", stride_cols = " << stride_cols
                << ", out_depth = " << num_filter;

        // If there is nothing to compute, return.
        if (out_shape.num_elements() == 0)
            return;

    }

private:
    void LayerSetUp(const TensorShape &ishape, const TensorShape &offset_shape,
                    const TensorShape &oshape) {
        channel_axis_ = 1;  // hard code channel axis
        const int first_spatial_axis = channel_axis_ + 1;
        const int num_axes = param_->kernel.dims() + 2;
        num_spatial_axes_ = num_axes - first_spatial_axis;
        is_1x1_ = true;
        for (int i = 0; i < param_->kernel.dims(); ++i) {
            is_1x1_ &= param_->kernel.dim_size(i) == 1 && param_->stride.dim_size(i) == 1 && param_->pad.dim_size(i) == 0;
            if (!is_1x1_) break;
        }

        // batch size
        num_ = ishape.dim_size(0);
        // number of input channels
        channels_ = ishape.dim_size(1);
        group_ = param_->num_group;
        conv_out_channels_ = param_->num_filter;
        conv_in_channels_ = channels_;
        bias_term_ = !param_->no_bias;
        kernel_dim_ = conv_in_channels_ / group_ * param_->kernel.dim_size(0) * param_->kernel.dim_size(1);
        weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;
        conv_out_spatial_dim_ = ProdShape(oshape, 2);
        col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
        output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
        // size of the column buffer used for storing im2col-ed pixels
        col_buffer_size_ = kernel_dim_ * group_ * conv_out_spatial_dim_;
        // input/output image size (#channels * height * width)
        input_dim_ = ProdShape(ishape, 1);
        input_offset_dim_ = ProdShape(offset_shape, 1);
        output_dim_ = ProdShape(oshape, 1);
        num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
        num_kernels_col2im_ = input_dim_;
    }

    //   DeformableConvolutionParam param_;
    int channel_axis_;       // channel axis of the input
    int channels_;           // number of channels of input image
    int num_spatial_axes_;   // number of spatial axes
    int num_;                // batch size
    int group_;              // number of groups
    int conv_out_channels_;  // number of output channels (num_filter)
    int conv_out_spatial_dim_;  // number of pixels of output images per channel
    int conv_in_channels_;  // number of input channels
    int kernel_dim_;     // number of input channels per group * kernel size
    int weight_offset_;  // number of output channels per group * kernel_dim_
    int col_offset_;
    int output_offset_;
    int col_buffer_size_;
    int input_dim_;
    int input_offset_dim_;
    int output_dim_;
    int num_kernels_im2col_;
    int num_kernels_col2im_;
    int num_groups;
    int deformable_group;
    bool bias_term_;  // has bias term?
    bool is_1x1_;

    std::vector<int32> strides_;
    std::vector<int32> rates_;
    Padding padding_;
    TensorFormat data_format_;
    DeformConvParam *param_;
};


//------------------------------------builder-----------------------------------------------------
//#define REGISTER_CPU(T)                                                 \
//  REGISTER_KERNEL_BUILDER(                                              \
//      Name("DeformConvOp").Device(DEVICE_CPU).TypeConstraint<T>("T"),   \
//      DeformConvOp<CPUDevice, T>);
//REGISTER_CPU(float);
//REGISTER_CPU(double);
//#undef REGISTER_CPU

#if GOOGLE_CUDA

#define REGISTER_GPU(T)                                                 \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("DeformConvOp").Device(DEVICE_GPU).TypeConstraint<T>("T"),   \
      DeformConvOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(double);
#undef REGISTER_GPU

#endif //GOOGLE_CUDA





// TF_CALL_GPU_NUMBER_TYPES(REGISTER);
//TF_CALL_float(REGISTER);
//TF_CALL_double(REGISTER);




