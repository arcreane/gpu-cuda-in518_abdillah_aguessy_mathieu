#pragma once

extern "C" void cuda_demo_dump(
    int grid_size,
    int block_size,
    int* out_blockdim,   // taille = grid*block
    int* out_threadIdx,  // taille = grid*block
    int* out_blockIdx,   // taille = grid*block
    int* out_globalIdx   // taille = grid*block
);