#include "RaylibView.h"

// ===== Raylib =====
#include "raylib.h"

// Qt
#include <QResizeEvent>
#include <QWindow>
#include <algorithm>




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

void RaylibView::resizeEvent(QResizeEvent* ev) {
    QWidget::resizeEvent(ev);
    reqW.store(ev->size().width());
    reqH.store(ev->size().height());
}

/* PARICULES - Methods */
void RaylibView::initParticlesCPU() {
    particles.clear();
    particles.reserve(maxParticles);

    int count = maxParticles;

    for (int i = 0; i < count; ++i) {
        Particle p;

        // spawn near the top center
        p.x    = 100.0f + static_cast<float>(i % 50) * 5.0f;
        p.y    = 50.0f + static_cast<float>((i / 50) % 10) * 5.0f;
        p.vx   = 30.0f * static_cast<float>((i % 10) - 5);
        p.vy   = -200.0f - static_cast<float>(i % 60);
        p.life = 3.0f + 0.01f * static_cast<float>(i);

        p.radius = 3.0f + static_cast<float>(i % 3);

        // simple color gradient
        p.r = static_cast<unsigned char>(150 + (i % 100));
        p.g = static_cast<unsigned char>(100 + (i % 120));
        p.b = static_cast<unsigned char>(200);
        p.a = 255;

        particles.push_back(p);
    }
}

void RaylibView::stepParticlesCPU(float dt) {
    for (std::size_t i = 0; i < particles.size(); ++i) {
        Particle& part = particles[i];

        // gravity
        part.vy += gravityY * dt;

        // damping
        part.vx *= damping;
        part.vy *= damping;

        // integrate
        part.x += part.vx * dt;
        part.y += part.vy * dt;

        // ground collision
        if (part.y > groundY) {
            part.y = groundY;
            part.vy *= -0.5f;
        }

        // life decrease
        part.life -= dt;
        if (part.life < 0.0f) {
            // respawn
            part.x    = 50.0f;
            part.y    = 50.0f;
            part.vx   = 40.0f * static_cast<float>((i % 12) - 6);
            part.vy   = -220.0f - static_cast<float>(i % 80);
            part.life = 4.0f;

            // quick color change
            part.r = static_cast<unsigned char>(200 + (i % 55));
            part.g = static_cast<unsigned char>(120 + (i % 80));
            part.b = 180;
            part.a = 255;
        }
    }
}

void RaylibView::stepParticlesGPU(float dt) {
#ifndef USE_CUDA
    (void)dt;
    // If CUDA is not available, fallback to CPU
    stepParticlesCPU(dt);
#else
    if (particles.empty()) return;

    if (!cudaInitialized) {
        cuda_particles_init(maxParticles);
        cudaInitialized = true;
    }

    int count = static_cast<int>(particles.size());

    // host -> device
    cuda_particles_upload(particles.data(), count);

    // physics on GPU
    cuda_particles_step(dt, gravityY, damping, groundY, count);

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

        int W = reqW.load(), H = reqH.load();
        if (W <= 0) W = 800; if (H <= 0) H = 450;

        // 1) Creer la fenetre Raylib
        InitWindow(W, H, "PartiQle Studio - CPU Particles");
        SetTargetFPS(60);

        curW.store(W);
        curH.store(H);
        groundY = static_cast<float>(H - 50);

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
                groundY = static_cast<float>(rh - 50);
            }

            // dt
            auto now = clock::now();
            float dt = std::chrono::duration<float>(now - lastTime).count();
            lastTime = now;
            if (dt > 0.05f) dt = 0.05f; // clamp

            // raccourcis touche G
#ifdef USE_CUDA
            if (IsKeyPressed(KEY_G)) {
                bool old = useGPU.load();
                useGPU.store(!old);
            }
#endif

            // simulation
#ifdef USE_CUDA
            if (useGPU.load()) {
                stepParticlesGPU(dt);
            } else {
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
            DrawLine(0, (int)groundY, curW.load(), (int)groundY, DARKGRAY);

            // draw particles
            for (const auto& part : particles) {
                Color c = {
                    part.r,
                    part.g,
                    part.b,
                    part.a
                };
                DrawCircleV(Vector2{ part.x, part.y }, part.radius, c);
            }

#ifdef USE_CUDA
            const char* mode = useGPU.load() ? "GPU" : "CPU";
            DrawText(TextFormat("Mode: %s (press G to toggle)", mode),
                     10, 10, 20, RAYWHITE);
#else
            // Exemple minimal de drawing (remarque : DrawText utilise la police par defaut)
            DrawText("Mode: CPU (CUDA not available)",
                     10, 10, 20, RAYWHITE);
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


