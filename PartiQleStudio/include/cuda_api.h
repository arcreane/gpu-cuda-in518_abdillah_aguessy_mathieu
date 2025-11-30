#pragma once

// Structure dune particule pour le kernel CUDA
struct Particle {
    float x, y;
    float vx, vy;
    float life;
    float radius;
    unsigned char r, g, b, a;
};

#ifdef USE_CUDA

// Alloue la mémoire GPU pour les particules (dimension max)
extern "C" void cuda_particles_init(int maxCount);

// Libre la mémoire GPU allouée pour les particules
extern "C" void cuda_particles_free();

// Copie les particules CPU vers la mémoire GPU
extern "C" void cuda_particles_upload(const Particle* hostParticles, int count);

// Evolution des particules sur le GPU
extern "C" void cuda_particles_step(float dt,
                                    float gravityY,
                                    float damping,
	                                float elasticity,
	                                float frictionCoeff,
                                    int   screenW,
                                    int   screenH,
                                    int   count);

// Récupère les particules GPU vers la mémoire CPU
extern "C" void cuda_particles_download(Particle* hostParticles, int count);

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

inline void cuda_particles_upload(const Particle*, int) {}

inline void cuda_particles_step(float, float, float, float, float, int, int, int) {}

inline void cuda_particles_download(Particle*, int) {}

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