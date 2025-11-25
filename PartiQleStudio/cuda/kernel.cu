#include <cuda_runtime.h>
#include <cstdio>
#include "cuda_api.h"

// --------------------------------------------------
// Kernels Demo
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
// DEMO // API C appelée depuis le code C++/Qt (déclarée dans cuda_api.h)
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

// --------------------------------------------------
// Buffer global particules
// --------------------------------------------------
static ParticleGPU* d_particles = nullptr;
static int          d_capacity = 0;

// --------------------------------------------------
// Kernel d'update particules
// --------------------------------------------------
__global__ void kernel_update_particles(ParticleGPU* p,
    int count,
    float dt,
    float gravityY,
    float damping,
    float groundY)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    ParticleGPU& part = p[i];

    // gravité
    part.vy += gravityY * dt;

    // frottement
    part.vx *= damping;
    part.vy *= damping;

    // intégration
    part.x += part.vx * dt;
    part.y += part.vy * dt;

    // rebond au sol
    if (part.y > groundY) {
        part.y = groundY;
        part.vy *= -0.5f; // rebond amorti
    }

    // décrément de "vie"
    part.life -= dt;
    if (part.life < 0.0f) {
        // simple "respawn" en haut
        part.x = 0.0f;
        part.y = 0.0f;
        part.vx = 20.0f * ((i % 10) - 5);
        part.vy = -200.0f;
        part.life = 5.0f;
    }
}

// --------------------------------------------------
// API exposée
// --------------------------------------------------
extern "C" void cuda_particles_init(int maxCount)
{
    if (d_particles) {
        CUDA_CHECK(cudaFree(d_particles));
        d_particles = nullptr;
        d_capacity = 0;
    }
    d_capacity = maxCount;
    CUDA_CHECK(cudaMalloc((void**)&d_particles,
        d_capacity * sizeof(ParticleGPU)));
}

extern "C" void cuda_particles_free()
{
    if (d_particles) {
        CUDA_CHECK(cudaFree(d_particles));
        d_particles = nullptr;
        d_capacity = 0;
    }
}

extern "C" void cuda_particles_upload(const ParticleGPU* hostParticles,
    int count)
{
    if (!d_particles || count > d_capacity) return;
    CUDA_CHECK(cudaMemcpy(d_particles,
        hostParticles,
        count * sizeof(ParticleGPU),
        cudaMemcpyHostToDevice));
}

extern "C" void cuda_particles_step(float dt,
    float gravityY,
    float damping,
    float groundY,
    int   count)
{
    if (!d_particles || count <= 0) return;

    int blockSize = 256;
    int gridSize = (count + blockSize - 1) / blockSize;

    kernel_update_particles << <gridSize, blockSize >> > (
        d_particles, count, dt, gravityY, damping, groundY
        );
    CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C" void cuda_particles_download(ParticleGPU* hostParticles,
    int count)
{
    if (!d_particles || count > d_capacity) return;

    CUDA_CHECK(cudaMemcpy(hostParticles,
        d_particles,
        count * sizeof(ParticleGPU),
        cudaMemcpyDeviceToHost));
}