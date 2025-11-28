#pragma once
#include <QWidget>
#include <QWindow>
#include <QVBoxLayout>
#include <atomic>
#include <thread>
#include <future>
#include <vector>

#include "cuda_api.h" //pour le define de Particle

class RaylibView : public QWidget {
    Q_OBJECT
public:
    explicit RaylibView(QWidget* parent = nullptr);
    ~RaylibView() override;

    // Exemple d’API pour piloter la couleur de fond si besoin
    void setClearColor(unsigned char r, unsigned char g, unsigned char b); //rgb

    void setUseGPU(bool enabled);
    bool isUsingGPU() const { return useGPU.load(); }

    // Gestion simulation (buttons)
    void setPaused(bool p);
    bool isPaused() const { return paused.load(); }

    void resetSimulation();           // init les particules
    void setParticleCount(int count); // spinParticles

    // Stats
    float fps() const { return lastFps.load(); }
    float frameTimeMs() const { return lastFrameMs.load(); }
    int   particleCount() const { return lastParticleCount.load(); }

protected:
    void resizeEvent(QResizeEvent* event) override;

private:
    void startRaylibThread();
    void stopRaylibThread();
    void embedHandleToQt(void* nativeHandle);

    void initParticlesCPU();
    void stepParticlesCPU(float dt);
    void stepParticlesGPU(float dt);

private:
    std::thread         rlThread;
    std::atomic<bool>   running{ false };

    // Taille demandée par Qt et taille réellement appliquée côté Raylib
    std::atomic<int>    reqW{ 800 }, reqH{ 450 };
    std::atomic<int>    curW{ 0 }, curH{ 0 };

    // Couleur de fond
    std::atomic<int>    clrR{ 245 }, clrG{ 245 }, clrB{ 245 };

    QWidget* containerWidget{ nullptr };
    QWindow* foreignWin{ nullptr };

    // particle simulation
    std::vector<Particle> particles;
    int   maxParticles = 100000;                // capacité max
	std::atomic<int> desiredParticles{ 1000 };  // spinParticles

    float gravityY = 0.1f;
    float damping = 0.999f;

	std::atomic<bool> useGPU{ false };
    std::atomic<bool> paused{ false };

    // Stats pour le panneau Qt
    std::atomic<float> lastFrameMs{ 0.0f };
    std::atomic<float> lastFps{ 0.0f };
    std::atomic<int>   lastParticleCount{ 0 };

#ifdef USE_CUDA
	bool cudaInitialized = false;
#endif

};
