#pragma once

// Structure dune particule pour le kernel CUDA
struct ParticleGPU {
    float x, y;
    float vx, vy;
    float life;
};

#ifdef USE_CUDA

// Alloue la mémoire GPU pour les particules (dimension max)
extern "C" void cuda_particles_init(int maxCount);

// Libre la mémoire GPU allouée pour les particules
extern "C" void cuda_particles_free();

// Copie les particules CPU vers la mémoire GPU
extern "C" void cuda_particles_upload(const ParticleGPU* hostParticles, int count);

// Evolution des particules sur le GPU
extern "C" void cuda_particles_step(float dt,
                                    float gravityY,
                                    float damping,
                                    float groundY,
                                    int   count);

// Récupère les particules GPU vers la mémoire CPU
extern "C" void cuda_particles_download(ParticleGPU* hostParticles, int count);

// Fonction démo
extern "C" void cuda_demo_dump(
    int grid_size,
    int block_size,
    int* out_blockdim,   // taille = grid*block
    int* out_threadIdx,  // taille = grid*block
    int* out_blockIdx,   // taille = grid*block
    int* out_globalIdx   // taille = grid*block
);

#else

inline void cuda_particles_init(int) {}
inline void cuda_particles_free() {}

inline void cuda_particles_upload(const ParticleGPU*, int) {}

inline void cuda_particles_step(float, float, float, float, int) {}

inline void cuda_particles_download(ParticleGPU*, int) {}

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