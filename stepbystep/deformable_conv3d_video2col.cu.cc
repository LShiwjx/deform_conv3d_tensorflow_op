#ifdef GOOGLE_CUDA

#define EIGEN_USE_GPU
#define EIGEN_USE_THREADS

#include "deformable_conv3d_video2col.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;

#define EIGEN_USE_GPU

// Define the CUDA kernel.
template <typename T>
__global__ void DeformableConv3dVideo2colCudaKernel(const int size, const T* in, T* out) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += blockDim.x * gridDim.x) {
        out[i] = 2 * ldg(in + i);
    }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct DeformableConv3dVideo2colFunctor<GPUDevice, T> {
    void operator()(const GPUDevice& d, int size, const T* in, T* out) {
        // Launch the cuda kernel.
        //
        // See core/util/cuda_kernel_helper.h for example of computing
        // block count and thread_per_block count.
        int block_count = 1024;
        int thread_per_block = 20;
        DeformableConv3dVideo2colCudaKernel<T>
                <<<block_count, thread_per_block, 0, d.stream()>>>(size, in, out);
    }
};

// Instantiate functors for the types of OpKernels registered.
typedef Eigen::GpuDevice GPUDevice;
template struct DeformableConv3dVideo2colFunctor<GPUDevice, float>;
template struct DeformableConv3dVideo2colFunctor<GPUDevice, int32>;

#endif  // GOOGLE_CUDA