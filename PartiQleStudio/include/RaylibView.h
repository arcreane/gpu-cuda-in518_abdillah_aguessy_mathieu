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
	void setElasticity(float e);      // spinElasticity
	void setFriction(float f);        // spinFriction
    void setVelocityMin(float vmin);
    void setVelocityMax(float vmax);
    void setMouseRadius(int radius);
    void setMouseForce(float force);

    // Debug overlays control
    void setShowMouseInfo(bool v);
    void setShowEngineInfo(bool v);
    void setShowPerfInfo(bool v);
	void setShowBoxsimInfo(bool v);
    bool isShowMouseInfo() const;
    bool isShowEngineInfo() const;
    bool isShowPerfInfo() const;
	bool isShowBoxsimInfo() const;

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

    void applyMouseForceCPU(float mouseX, float mouseY, float velX, float velY, int mode);

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

    std::atomic<float> frictionCoeff{ 0.0f };   // spinFriction
    std::atomic<float> elasticity{ 0.9f };      // spinElasticity
    std::atomic<float> velocityMin{ -50.0f };   // Vmin
    std::atomic<float> velocityMax{ 50.0f };    // Vmax

    std::atomic<int>   mouseRadius{ 100 };      // Rayon d'action (pixels)
    std::atomic<float> mouseForceScale{ 0.5f }; // Multiplicateur de force

	std::atomic<bool> useGPU{ false };
    std::atomic<bool> paused{ true };

    // Debug overlays flags
    std::atomic<bool> showMouseInfo{ false };
    std::atomic<bool> showEngineInfo{ false };
    std::atomic<bool> showPerfInfo{ false };
	std::atomic<bool> showBoxsimInfo{ false };


    // Stats pour le panneau Qt
    std::atomic<float> lastFrameMs{ 0.0f };
    std::atomic<float> lastFps{ 0.0f };
    std::atomic<int>   lastParticleCount{ 0 };

#ifdef USE_CUDA
	bool cudaInitialized = false;
    int lastUploadedCount = 0;
#endif

};
