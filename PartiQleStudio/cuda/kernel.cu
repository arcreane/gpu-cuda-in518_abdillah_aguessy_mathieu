// kernel.cu — CUDA only (aucun main, aucune dépendance Qt/Raylib)
#include <cuda_runtime.h>
#include <cstdio>

// --------------------------------------------------
// Kernels
// --------------------------------------------------
__global__ void kernel_idx(int* a) {a[threadIdx.x + blockDim.x * blockIdx.x] = threadIdx.x + blockDim.x * blockIdx.x;}
__global__ void kernel_blockdim(int* a) {a[threadIdx.x + blockDim.x * blockIdx.x] = blockDim.x;}
__global__ void kernel_threadIdx(int* a) {a[threadIdx.x + blockDim.x * blockIdx.x] = threadIdx.x;}
__global__ void kernel_blockIdx(int* a) {a[threadIdx.x + blockDim.x * blockIdx.x] = blockIdx.x;}

// --------------------------------------------------
// Helper macro
// --------------------------------------------------
#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do {                                   \
    cudaError_t _e = (call);                                    \
    if (_e != cudaSuccess) {                                    \
        fprintf(stderr, "CUDA error %s:%d: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(_e));    \
    }                                                           \
} while(0)
#endif

// --------------------------------------------------
// API C appelée depuis le code C++/Qt (déclarée dans cuda_api.h)
// Remplit 4 buffers hôte de taille grid_size*block_size
// --------------------------------------------------
extern "C" void cuda_demo_dump(
    int grid_size,
    int block_size,
    int* out_blockdim,   // [size]
    int* out_threadIdx,  // [size]
    int* out_blockIdx,   // [size]
    int* out_globalIdx   // [size]
) {
    const int size = grid_size * block_size;

    int* d_a = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_a, size * sizeof(int)));

    // blockDim.x
    kernel_blockdim << <grid_size, block_size >> > (d_a);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(out_blockdim, d_a, size * sizeof(int), cudaMemcpyDeviceToHost));

    // threadIdx.x
    kernel_threadIdx << <grid_size, block_size >> > (d_a);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(out_threadIdx, d_a, size * sizeof(int), cudaMemcpyDeviceToHost));

    // blockIdx.x
    kernel_blockIdx << <grid_size, block_size >> > (d_a);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(out_blockIdx, d_a, size * sizeof(int), cudaMemcpyDeviceToHost));

    // global index
    kernel_idx << <grid_size, block_size >> > (d_a);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(out_globalIdx, d_a, size * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_a));
}
