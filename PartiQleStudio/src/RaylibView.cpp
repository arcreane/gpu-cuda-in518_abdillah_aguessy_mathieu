#include "RaylibView.h"

// ===== Raylib =====
#include "raylib.h"

// Qt
#include <QResizeEvent>
#include <QWindow>

#include <algorithm>
#include <chrono>
#include <cstdlib>

static float frand(float a, float b) {
    return a + (b - a) * (float)rand() / (float)RAND_MAX;
}

RaylibView::RaylibView(QWidget* parent) : QWidget(parent) {
    setAttribute(Qt::WA_NativeWindow, false);
    setAttribute(Qt::WA_DontCreateNativeAncestors, true);
    startRaylibThread();
}

RaylibView::~RaylibView() {
    stopRaylibThread();
}

void RaylibView::setClearColor(unsigned char r, unsigned char g, unsigned char b) {
    clrR.store(static_cast<int>(r));
    clrG.store(static_cast<int>(g));
    clrB.store(static_cast<int>(b));
}

void RaylibView::setUseGPU(bool enabled) {
#ifdef USE_CUDA
    useGPU.store(enabled, std::memory_order_relaxed);
#else
    (void)enabled;
    useGPU.store(false, std::memory_order_relaxed);
#endif
}

void RaylibView::setPaused(bool p) {
    paused.store(p, std::memory_order_relaxed);
}

void RaylibView::resetSimulation() {
    initParticlesCPU();
}


void RaylibView::setParticleCount(int count) {
    if (count < 0) count = 0;
    if (count > maxParticles) count = maxParticles;
    desiredParticles.store(count, std::memory_order_relaxed);
    initParticlesCPU();
}

void RaylibView::resizeEvent(QResizeEvent* ev) {
    QWidget::resizeEvent(ev);
    reqW.store(ev->size().width());
    reqH.store(ev->size().height());
}

/* PARICULES CPU - Methods */
void RaylibView::initParticlesCPU() {
    int width = curW.load();
    int height = curH.load();
    if (width <= 0)  width = 800;
    if (height <= 0) height = 450;

    int count = desiredParticles.load();
    if (count <= 0) count = 0;
    if (count > maxParticles) count = maxParticles;

    particles.clear();
    particles.reserve(maxParticles);

    for (int i = 0; i < count; ++i) {
        Particle p;

        p.radius = frand(3.0f, 6.0f);

        // spawn near the top center
        p.x = frand(p.radius, width - p.radius);
        p.y = frand(p.radius, height - p.radius);
        
        // v init légère et vers le haut
        /*p.vx = frand(-50.0f, 50.0f);
        p.vy = frand(-80.0f, -20.0f);*/
        p.vx = frand(-15.0f, 15.0f);
        p.vy = frand(-15.0f, 15.0f);

        p.life = 1.0f;


        // simple color gradient
        p.r = (unsigned char)(100 + (i * 13) % 155);
        p.g = (unsigned char)(100 + (i * 29) % 155);
        p.b = (unsigned char)(180 + (i * 7) % 75);
        p.a = 255;

        particles.push_back(p);
    }
}

void RaylibView::stepParticlesCPU(float dt) {
    int width = curW.load();
    int height = curH.load();
    if (width <= 0) width = 800;
    if (height <= 0) height = 450;

    const float bounce = 0.9f;

    for (auto& p : particles) {
        // gravité
        p.vy += gravityY * dt;

        // frottement global
        p.vx *= damping;
        p.vy *= damping;

        // intégration
        p.x += p.vx * dt;
        p.y += p.vy * dt;

        float r = p.radius;

        // collision horizontale
        if (p.x < r) {
            p.x = r;
            p.vx = -p.vx * bounce;
        }
        else if (p.x > width - r) {
            p.x = width - r;
            p.vx = -p.vx * bounce;
        }

        // collision verticale
        if (p.y < r) {
            p.y = r;
            p.vy = -p.vy * bounce;
        }
        else if (p.y > height - r) {
            p.y = height - r;
            p.vy = -p.vy * bounce;
        }
    }
}


/* PARICULES GPU - Methods */
void RaylibView::stepParticlesGPU(float dt) {
#ifndef USE_CUDA
    (void)dt;
    stepParticlesCPU(dt);
#else
    if (particles.empty()) return;

    if (!cudaInitialized) {
        cuda_particles_init(maxParticles);
        cudaInitialized = true;
    }

    int count = static_cast<int>(particles.size());
    int width = curW.load();
    int height = curH.load();
    if (width <= 0) width = 800;
    if (height <= 0) height = 450;

    // host -> device
    cuda_particles_upload(particles.data(), count);

    // physics on GPU
    cuda_particles_step(dt, gravityY, damping, width, height, count);

    // device -> host
    cuda_particles_download(particles.data(), count);
#endif
}

/* Raylib fonctionnement thread */
void RaylibView::startRaylibThread() {
    if (running) return;
    running = true;

    // future/promise pour recuperer le handle natif de la fenetre Raylib
    auto p = std::make_shared<std::promise<void*>>();
    auto f = p->get_future();

    rlThread = std::thread([this, p]() {
        // Eviter que la touche ESC ne ferme la fenetre
        SetExitKey(0);

        int W = reqW.load();
        int H = reqH.load();
        if (W <= 0) W = 800; 
        if (H <= 0) H = 450;

        // 1) Creer la fenetre Raylib
        InitWindow(W, H, "PartiQle Studio - Particles");
        SetTargetFPS(60);

        curW.store(W);
        curH.store(H);

		// Initialiser les particules
        initParticlesCPU();

        // 2) Recuperer le handle natif et le transmettre a Qt
        void* native = GetWindowHandle();
        p->set_value(native);

        using clock = std::chrono::steady_clock;
        auto lastTime = clock::now();

        // 3) rendu raylib
        while (running.load() && !WindowShouldClose()) {
            // handle resize
            int rw = reqW.load();
            int rh = reqH.load();
            if (rw > 0 && rh > 0 && (rw != curW.load() || rh != curH.load())) {
                SetWindowSize(rw, rh);
                curW.store(rw);
                curH.store(rh);
            }

            // dt
            auto now = clock::now();
            float dt = std::chrono::duration<float>(now - lastTime).count();
            lastTime = now;
            if (dt > 0.05f) dt = 0.05f; // clamp

            // raccourcis touche G
#ifdef USE_CUDA
            bool gpuNow = useGPU.load(std::memory_order_relaxed);
            SetWindowTitle(gpuNow ? "PartiQle Studio - GPU mode"
                : "PartiQle Studio - CPU mode");
#endif

            // simulation
#ifdef USE_CUDA
            if (useGPU.load(std::memory_order_relaxed)) {
                stepParticlesGPU(dt);
            }
            else {
                stepParticlesCPU(dt);
            }
#else
            stepParticlesCPU(dt);
#endif

            // set des couleurs (rgb)
            int r = clrR.load();
            int g = clrG.load();
            int b = clrB.load();
            Color bg = { (unsigned char)r, (unsigned char)g, (unsigned char)b, 255 };

            BeginDrawing();
            ClearBackground(bg);

			// Dessin des particules
            for (const auto& part : particles) {
                Color c = { part.r, part.g, part.b, part.a };
                DrawCircleV(Vector2{ part.x, part.y }, part.radius, c);
            }

#ifdef USE_CUDA
            const char* mode = useGPU.load(std::memory_order_relaxed) ? "GPU" : "CPU";
            DrawText(TextFormat("Mode: %s", mode),
                10, 10, 20, GREEN);
            if (useGPU.load())
                DrawText("GPU ACTIVE", 10, 40, 20, GREEN);
            else
                DrawText("CPU ACTIVE", 10, 40, 20, RED);

#else
            // Exemple minimal de drawing (remarque : DrawText utilise la police par defaut)
            DrawText("Mode: CPU (CUDA not available)",
                10, 10, 20, DARKGRAY);
#endif

            EndDrawing();
        }

#ifdef USE_CUDA
        if (cudaInitialized) {
            cuda_particles_free();
            cudaInitialized = false;
        }
#endif

        // Fermer proprement la fenetre Raylib
        CloseWindow();
        running.store(false);
    });

    // Recuperer le handle natif et l'embedder dans l'UI Qt (sur le thread Qt)
    void* native = f.get();
    embedHandleToQt(native);
}

void RaylibView::stopRaylibThread() {
    if (!running.load()) return;
    running.store(false);
    if (rlThread.joinable()) rlThread.join();

    // Detruire la fenetre embed si elle existe
    if (containerWidget) {
        delete containerWidget;
        containerWidget = nullptr;
        foreignWin = nullptr;
    }
}

void RaylibView::embedHandleToQt(void* nativeHandle) {
    if (!nativeHandle) return;

    // Convert native handle to Qt WId and create a QWindow wrapper
    WId wid = reinterpret_cast<WId>(nativeHandle);
    foreignWin = QWindow::fromWinId(wid);
    if (!foreignWin) return;

    // Create a QWidget that can host the QWindow
    containerWidget = QWidget::createWindowContainer(foreignWin, this);
    containerWidget->setFocusPolicy(Qt::StrongFocus);
    containerWidget->setMinimumSize(reqW.load(), reqH.load());

    // Insert the container into this widget's layout (if none, make it fill)
    if (auto* layout = this->layout()) {
        layout->addWidget(containerWidget);
    } else {
        auto* l = new QVBoxLayout(this);
        l->setContentsMargins(0, 0, 0, 0);
        l->addWidget(containerWidget);
        setLayout(l);
    }
}


