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
    REGISTER_OP("DeformConv3dOp")
            .Input("x: T") //T是类型
            .Input("filter: T")
            .Input("offset: T")
            .Output("output: T")
            .Input("strides_shape: T")
            .Input("dilations_shape: T") //dilation rates
            .Attr("deformable_groups: int64")
            .Attr(GetPaddingAttrString()) //得到到pad类型
            .Attr("T: {float, double}")
            .SetShapeFn([](InferenceContext *c) {
                //TODO:shilei
                ShapeHandle input_shape;
                TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 5, &input_shape));
                ShapeHandle filter_shape;
                TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &filter_shape));//withrank 确定维数正确,5维
                ShapeHandle offset_shape;
                TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 5, &offset_shape));
                ShapeHandle strides_shape;
                TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 3, &strides_shape));
                ShapeHandle dilations_shape;
                TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 3, &dilations_shape));

                Padding padding;
                TF_RETURN_IF_ERROR(c->GetAttr("padding", &padding));
                int64 group;
                TF_RETURN_IF_ERROR(c->GetAttr("deformable_groups", &group));

                DimensionHandle batch_size_dim = c->Dim(input_shape, 0);//nclhw
                DimensionHandle in_channels_dim = c->Dim(input_shape, 1);
                DimensionHandle in_lens_dim = c->Dim(input_shape, 2);
                DimensionHandle in_rows_dim = c->Dim(input_shape, 3);
                DimensionHandle in_cols_dim = c->Dim(input_shape, 4);
                //groups lhw out大小由输出定，out由卷积核大小定
                DimensionHandle offset_groups_dim = c->Dim(offset_shape, 0);
                DimensionHandle offset_lens_dim = c->Dim(offset_shape, 1);
                DimensionHandle offset_cols_dim = c->Dim(offset_shape, 2);
                DimensionHandle offset_rows_dim = c->Dim(offset_shape, 3);
                DimensionHandle offset_depth_dim = c->Dim(offset_shape, 4);
                //clhw
                DimensionHandle filter_channels_dim = c->Dim(filter_shape, 0);
                DimensionHandle filter_lens_dim = c->Dim(filter_shape, 1);
                DimensionHandle filter_cols_dim = c->Dim(filter_shape, 2);
                DimensionHandle filter_rows_dim = c->Dim(filter_shape, 3);

                DimensionHandle strides_lens_dim = c->Dim(strides_shape, 0);
                DimensionHandle strides_cols_dim = c->Dim(strides_shape, 1);
                DimensionHandle strides_rows_dim = c->Dim(strides_shape, 2);

                DimensionHandle dilations_lens_dim = c->Dim(strides_shape, 0);
                DimensionHandle dilations_cols_dim = c->Dim(strides_shape, 1);
                DimensionHandle dilations_rows_dim = c->Dim(strides_shape, 2);

                auto filter_lens = c->Value(filter_lens_dim);
                auto filter_cols = c->Value(filter_cols_dim);
                auto filter_rows = c->Value(filter_rows_dim);
                auto filter_channels = c->Value(filter_channels_dim);

                auto in_lens = c->Value(in_lens_dim);
                auto in_cols = c->Value(in_cols_dim);
                auto in_rows = c->Value(in_rows_dim);
                auto in_channels = c->Value(in_channels_dim);

                auto offset_groups = c->Value(offset_groups_dim);
                auto offset_lens = c->Value(offset_lens_dim);
                auto offset_cols = c->Value(offset_cols_dim);
                auto offset_rows = c->Value(offset_rows_dim);
                auto offset_depth = c->Value(offset_depth_dim);

                auto strides_lens = c->Value(strides_lens_dim);
                auto strides_cols = c->Value(strides_cols_dim);
                auto strides_rows = c->Value(strides_rows_dim);

                auto dilations_lens = c->Value(dilations_lens_dim);
                auto dilations_cols = c->Value(dilations_cols_dim);
                auto dilations_rows = c->Value(dilations_rows_dim);


                //根据pad计算出书大小
                int64 output_rows, output_cols, output_lens;
                int64 padding_after;
                TF_RETURN_IF_ERROR(GetWindowedOutputSize(
                        in_lens, filter_lens, strides_lens, padding, &output_lens, &padding_after));
                TF_RETURN_IF_ERROR(GetWindowedOutputSize(
                        in_cols, filter_cols, strides_cols, padding, &output_cols, &padding_after));
                TF_RETURN_IF_ERROR(GetWindowedOutputSize(
                        in_rows, filter_rows, strides_rows, padding, &output_rows, &padding_after));
                //测试偏执的维度，大小应该与输出相同，深度应该是卷积核大小×3 lllhhhwww
                if (offset_lens != output_lens || offset_cols != output_cols || offset_rows != output_rows
                    || offset_depth != 3 * filter_lens * filter_cols * filter_rows) {
                    return errors::InvalidArgument(
                            "Deformconv requires the offset compatible with filter, but got: ",
                            c->DebugString(offset_shape));
                }

                int64 out_channels = in_channels * filter_channels;
                ShapeHandle output_shape = c->MakeShape(
                        {batch_size_dim, out_channels, output_lens, output_cols, output_rows});
                c->set_output(0, output_shape);
                return Status::OK();
            })
            .Doc(R"doc(??)doc");
}

//-----------------------------------------CPUDevice kernel implement----------------------------------
template<typename DType>
struct deformable_im2col<CPUDevice, DType> {
    void operator()(const Device &d,
                    const DType *data_im, const DType *data_offset,
                    const TensorShape &im_shape, const TensorShape &col_shape, const TensorShape &kernel_shape,
                    const Tensor &pad, const Tensor &stride, const Tensor &dilation,
                    const int deformable_group, DType *data_col) {};
};

void cpumatmul() {};
//-------------------------------------------------------计算-----------------------------------------------
namespace {
    template<typename T>
    perftools::gputools::DeviceMemory<T> AsDeviceMemory(const T *cuda_memory) {
        perftools::gputools::DeviceMemoryBase wrapped(const_cast<T *>(cuda_memory));
        perftools::gputools::DeviceMemory<T> typed(wrapped);
        return typed;
    }
}
namespace functor {
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
        padding_ = context->GetAttr("padding", &padding_);
        deformable_group_ = context->GetAttr("deformable_group", &deformable_group_);
    }

    void Compute(OpKernelContext *context) override {
        //nclhw
        const Tensor &input = context->input(0);
        const TensorShape &ishape = input.shape();
        //clhw
        const Tensor &filter = context->input(1);
        const TensorShape &filter_shape = filter.shape();
        //glhwd
        const Tensor &offset = context->input(2);
        const TensorShape &offset_shape = offset.shape();

        const Tensor &strides = context->input(3);

        const Tensor &dilations = context->input(4);


        int64 out_rows = 0, out_cols = 0, out_lens = 0, pad_lens = 0, pad_rows = 0, pad_cols = 0;
        //TODO:计算padding厚的大小
        OP_REQUIRES_OK(context,
                       GetWindowedOutputSize(ishape.dim_size(2), filter_shape.dim_size(1), strides.dim_size(0),
                                             padding_, &out_lens, &pad_lens));
        OP_REQUIRES_OK(context,
                       GetWindowedOutputSize(ishape.dim_size(3), filter_shape.dim_size(2), strides.dim_size(1),
                                             padding_, &out_cols, &pad_cols));
        OP_REQUIRES_OK(context,
                       GetWindowedOutputSize(ishape.dim_size(4), filter_shape.dim_size(3), strides.dim_size(2),
                                             padding_, &out_rows, &pad_rows));

        TensorShape pad({static_cast<int64>(pad_lens), static_cast<int64>(pad_rows), static_cast<int64>(pad_cols)});
        TensorShape kernels({filter_shape.dim_size(1), filter_shape.dim_size(2), filter_shape.dim_size(3)});
        TensorShape out_shape(
                {input.dim_size(0), input.dim_size(1) * filter.dim_size(0), out_lens, out_cols, out_rows});

        //滤波器
        const T *filter_4d_ptr = filter.template flat<T>().data();
        //输出
        Tensor *output_5d = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output_5d));
        T *output_5d_ptr = output_5d->template flat<T>().data();
        //展开的向量
        auto col_buf_6d_shape = TensorShape(
                {input.dim_size(0), input.dim_size(1) * filter.dim_size(0), out_lens, out_cols, out_rows,
                 filter.dim_size(0) * filter.dim_size(1) * filter.dim_size(2)});
        Tensor col_buffer_6d = nullptr;
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, col_buf_6d_shape, &col_buffer_6d));
//        OP_REQUIRES_OK(context, context->allocate_output(0, col_buf_6d_shape, &col_buffer_6d));//TODO
        auto col_buffer_6d_ptr = col_buffer_6d.template flat<T>().data();
        //输入
        auto in_data_ptr = input.template flat<T>().data();
        //偏执
        auto offset_ptr = offset.template flat<T>().data();

        const Device &d = context->eigen_device<Device>();
        //一次处理一个batch
        for (int n = 0; n < input.dim_size(0); ++n) {
            // 将图片deformable并展开
            deformable_im2col<Device, T>()(d, (in_data_ptr + n * ProdShape(ishape, 1)),
                                           (offset_ptr + n * ProdShape(offset_shape, 0)),
                                           ishape, col_buf_6d_shape, filter_shape,
                                           pad, strides, dilations, deformable_group_,
                                           col_buffer_6d_ptr);
            output_5d_ptr=col_buffer_6d_ptr;
//            functor::LaunchBatchMatMul<T>::Launch(context,
//                                                  filter_shape, col_buf_6d_shape,
//                                                  filter_4d_ptr, col_buffer_6d_ptr,
//                                                  false, false,
//                                                  output_5d_ptr + n * ProdShape(out_shape, 1));
        }


        // If there is nothing to compute, return.
        if (out_shape.num_elements() == 0)
            return;

    }

private:
    void LayerSetUp() {}

    Padding padding_;
    int64 deformable_group_;
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




