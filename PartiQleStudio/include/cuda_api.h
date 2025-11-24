#pragma once

#ifdef USE_CUDA

extern "C" void cuda_demo_dump(
    int grid_size,
    int block_size,
    int* out_blockdim,   // taille = grid*block
    int* out_threadIdx,  // taille = grid*block
    int* out_blockIdx,   // taille = grid*block
    int* out_globalIdx   // taille = grid*block
);

#else

inline void cuda_demo_dump(
    int /*grid_size*/,
    int /*block_size*/,
    int* /*out_blockdim*/,
    int* /*out_threadIdx*/,
    int* /*out_blockIdx*/,
    int* /*out_globalIdx*/
) {
    // On ne fait rien : version "no-op"
}

#endif