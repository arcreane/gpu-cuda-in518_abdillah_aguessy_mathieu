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
    std::atomic<int>    clrR{ 20 }, clrG{ 20 }, clrB{ 40 };

    QWidget* containerWidget{ nullptr };
    QWindow* foreignWin{ nullptr };

    // particle simulation
    std::vector<Particle> particles;
    int   maxParticles = 10000;
    float gravityY = 300.0f;
    float damping = 0.99f;
    float groundY = 420.0f;

	std::atomic<bool>   useGPU{ false };

#ifdef USE_CUDA
	bool cudaInitialized = false;
#endif

};
