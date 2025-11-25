#pragma once
#include <QWidget>
#include <QWindow>
#include <QVBoxLayout>
#include <atomic>
#include <thread>
#include <future>
#include <vector>

struct CpuParticle {
    float x, y;
    float vx, vy;
	float radius;
	unsigned char r, g, b, a;
};

class RaylibView : public QWidget {
    Q_OBJECT
public:
    explicit RaylibView(QWidget* parent = nullptr);
    ~RaylibView() override;

    // Exemple d’API pour piloter la couleur de fond si besoin
    void setClearColor(unsigned char r, unsigned char g, unsigned char b); //rgb

protected:
    void resizeEvent(QResizeEvent* event) override;

private:
    void startRaylibThread();
    void stopRaylibThread();
    void embedHandleToQt(void* nativeHandle);

	void initParticles(int count, int width, int height);
	void updateParticlesCPU(float dt, int width, int height);

private:
    std::thread         rlThread;
    std::atomic<bool>   running{ false };

    // Taille demandée par Qt et taille réellement appliquée côté Raylib
    std::atomic<int>    reqW{ 800 }, reqH{ 450 };
    std::atomic<int>    curW{ 0 }, curH{ 0 };

    // Couleur de fond (RAYWHITE par défaut)
    std::atomic<int>    clrR{ 245 }, clrG{ 245 }, clrB{ 245 };

    QWidget* containerWidget{ nullptr };
    QWindow* foreignWin{ nullptr };

	std::vector<CpuParticle> particles_;
    float gravity_ = 400.0f;
    float damping_ = 0.999f;
};
