#include <cuda_runtime.h>
#include <cstdio>
#include <math.h>
#include "cuda_api.h"

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
// Buffers Globaux
// --------------------------------------------------
static const int GRID_RES_X = 64;
static const int GRID_RES_Y = 32;
static const int MAX_CELLS = GRID_RES_X * GRID_RES_Y;

static Particle* d_particles = nullptr;
static int       d_capacity = 0;
static int* d_cellHead = nullptr;
static int* d_next = nullptr;

// --------------------------------------------------
// Intégration & Murs
// --------------------------------------------------
__global__ void kernel_integrate(
    Particle * p, int count, float dt, float gravityY,
    float damping, float elasticity, int screenW, int screenH)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    Particle& part = p[i];

    // Gravité
    part.vy += gravityY * dt;

    // Frottement
    part.vx *= damping;
    part.vy *= damping;

    // Intégration
    part.x += part.vx * dt;
    part.y += part.vy * dt;

    // Collisions avec les murs
    float r = part.radius;
    if (part.x < r) { part.x = r; part.vx *= -elasticity; }
    else if (part.x > screenW - r) { part.x = screenW - r; part.vx *= -elasticity; }

    if (part.y < r) { part.y = r; part.vy *= -elasticity; }
    else if (part.y > screenH - r) { part.y = screenH - r; part.vy *= -elasticity; }
}

// --------------------------------------------------
// Grille Spatiale
// --------------------------------------------------
__global__ void kernel_clear_grid(int* cellHead, int numCells)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numCells) {
        cellHead[i] = -1;
    }
}

__global__ void kernel_build_grid(
    Particle* particles, int count, int* cellHead, int* next, int screenW, int screenH)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    Particle& p = particles[i];
    float cellSizeX = (float)screenW / GRID_RES_X;
    float cellSizeY = (float)screenH / GRID_RES_Y;

    int cx = (int)(p.x / cellSizeX);
    int cy = (int)(p.y / cellSizeY);

    if (cx < 0) cx = 0; else if (cx >= GRID_RES_X) cx = GRID_RES_X - 1;
    if (cy < 0) cy = 0; else if (cy >= GRID_RES_Y) cy = GRID_RES_Y - 1;

    int cellIndex = cy * GRID_RES_X + cx;
    int oldHead = atomicExch(&cellHead[cellIndex], i);
    next[i] = oldHead;
}

// --------------------------------------------------
// Résolution des Collisions (Solveur Itératif)
// --------------------------------------------------
__global__ void kernel_solve_collisions(
    Particle* particles, int count, int* cellHead, int* next,
    float elasticity, int screenW, int screenH)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    Particle pi = particles[i];
    float cellSizeX = (float)screenW / GRID_RES_X;
    float cellSizeY = (float)screenH / GRID_RES_Y;

    int cx = (int)(pi.x / cellSizeX);
    int cy = (int)(pi.y / cellSizeY);

    if (cx < 0) cx = 0; else if (cx >= GRID_RES_X) cx = GRID_RES_X - 1;
    if (cy < 0) cy = 0; else if (cy >= GRID_RES_Y) cy = GRID_RES_Y - 1;

    for (int oy = -1; oy <= 1; ++oy) {
        for (int ox = -1; ox <= 1; ++ox) {
            int nx = cx + ox;
            int ny = cy + oy;

            if (nx < 0 || nx >= GRID_RES_X || ny < 0 || ny >= GRID_RES_Y) continue;

            int cellIndex = ny * GRID_RES_X + nx;
            int j = cellHead[cellIndex];

            while (j != -1) {
                if (i >= j) { // Traiter chaque paire une seule fois et éviter l'auto-collision
                    j = next[j];
                    continue;
                }

                Particle pj = particles[j];
                float dx = pj.x - pi.x;
                float dy = pj.y - pi.y;
                float dist2 = dx * dx + dy * dy;
                float rSum = pi.radius + pj.radius;

                if (dist2 > 1e-6f && dist2 < rSum * rSum) {
                    float dist = sqrtf(dist2);
                    float nxn = dx / dist;
                    float nyn = dy / dist;

                    // 1. Séparation (Position-Based Correction)
                    float overlap = (rSum - dist) * 0.5f; // 0.5 pour répartir la correction
                    atomicAdd(&particles[i].x, -overlap * nxn);
                    atomicAdd(&particles[i].y, -overlap * nyn);
                    atomicAdd(&particles[j].x, overlap * nxn);
                    atomicAdd(&particles[j].y, overlap * nyn);

                    // 2. Impulsion (Réponse à la collision)
                    float rvx = pj.vx - pi.vx;
                    float rvy = pj.vy - pi.vy;
                    float vn = rvx * nxn + rvy * nyn;

                    if (vn < 0) {
                        float impulse = -(1.0f + elasticity) * vn * 0.5f;
                        atomicAdd(&particles[i].vx, -impulse * nxn);
                        atomicAdd(&particles[i].vy, -impulse * nyn);
                        atomicAdd(&particles[j].vx, impulse * nxn);
                        atomicAdd(&particles[j].vy, impulse * nyn);
                    }
                }
                j = next[j];
            }
        }
    }
}

// --------------------------------------------------
// API C Externe
// --------------------------------------------------
extern "C" void cuda_particles_init(int maxCount) {
    if (d_particles) cudaFree(d_particles);
    if (d_cellHead) cudaFree(d_cellHead);
    if (d_next) cudaFree(d_next);

    d_capacity = maxCount;
    if (d_capacity > 0) {
        CUDA_CHECK(cudaMalloc(&d_particles, d_capacity * sizeof(Particle)));
        CUDA_CHECK(cudaMalloc(&d_next, d_capacity * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_cellHead, MAX_CELLS * sizeof(int)));
    }
}

extern "C" void cuda_particles_free() {
    cudaFree(d_particles);
    cudaFree(d_cellHead);
    cudaFree(d_next);
    d_particles = nullptr;
    d_cellHead = nullptr;
    d_next = nullptr;
    d_capacity = 0;
}

extern "C" void cuda_particles_upload(const Particle* hostParticles, int count) {
    if (!d_particles || count <= 0 || count > d_capacity) return;
    CUDA_CHECK(cudaMemcpy(d_particles, hostParticles, count * sizeof(Particle), cudaMemcpyHostToDevice));
}

extern "C" void cuda_particles_step(float dt,
    float gravityY, float baseDamping, float elasticity, float frictionCoeff,
    int screenW, int screenH, int count)
{
    if (!d_particles || count <= 0) return;

    float perStepDamp = baseDamping * (1.0f - frictionCoeff * dt);
    if (perStepDamp < 0.0f) perStepDamp = 0.0f;
    if (perStepDamp > 1.0f) perStepDamp = 1.0f;

    int blockSize = 256;
    int gridParticles = (count + blockSize - 1) / blockSize;
    int gridCells = (MAX_CELLS + blockSize - 1) / blockSize;

    // 1. Intégrer la physique (gravité, frottement, murs)
    kernel_integrate << <gridParticles, blockSize >> > (
        d_particles, count, dt, gravityY, perStepDamp, elasticity, screenW, screenH);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 2. Résoudre les collisions de manière itérative
    const int solverIterations = 4; // Standard pour un bon équilibre stabilité/performance
    for (int i = 0; i < solverIterations; ++i) {
        // a. Construire la grille
        kernel_clear_grid << <gridCells, blockSize >> > (d_cellHead, MAX_CELLS);
        kernel_build_grid << <gridParticles, blockSize >> > (
            d_particles, count, d_cellHead, d_next, screenW, screenH);
        CUDA_CHECK(cudaDeviceSynchronize());

        // b. Résoudre les collisions (séparation + impulsion)
        kernel_solve_collisions << <gridParticles, blockSize >> > (
            d_particles, count, d_cellHead, d_next, elasticity, screenW, screenH);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

extern "C" void cuda_particles_download(Particle* hostParticles, int count) {
    if (!d_particles || count <= 0) return;
    CUDA_CHECK(cudaMemcpy(hostParticles, d_particles, count * sizeof(Particle), cudaMemcpyDeviceToHost));
}

// La fonction cuda_demo_dump n'est plus nécessaire pour la simulation
extern "C" void cuda_demo_dump(int, int, int*, int*, int*, int*) {}