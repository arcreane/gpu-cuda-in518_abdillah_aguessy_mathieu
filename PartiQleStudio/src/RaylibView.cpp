#include "RaylibView.h"

// ===== Raylib =====
#include "raylib.h"

// Qt
#include <QResizeEvent>
#include <QWindow>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cmath>

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

void RaylibView::setElasticity(float e) {
    elasticity.store(e, std::memory_order_relaxed);
}

void RaylibView::setFriction(float f) {
    frictionCoeff.store(f, std::memory_order_relaxed);
}

void RaylibView::setVelocityMin(float vmin) {
    velocityMin.store(vmin, std::memory_order_relaxed);
}

void RaylibView::setVelocityMax(float vmax) {
    velocityMax.store(vmax, std::memory_order_relaxed);
}

void RaylibView::resetSimulation() {
    paused.store(true, std::memory_order_relaxed);
    particles.clear();
    lastParticleCount.store(0, std::memory_order_relaxed);
#ifdef USE_CUDA
    lastUploadedCount = 0;
#endif
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

    float vmin = velocityMin.load(std::memory_order_relaxed);
    float vmax = velocityMax.load(std::memory_order_relaxed);

    particles.clear();
    particles.reserve(maxParticles);

    for (int i = 0; i < count; ++i) {
        Particle p;

        p.radius = frand(3.0f, 6.0f);

        // spawn near the top center
        p.x = frand(p.radius, width - p.radius);
        p.y = frand(p.radius, height - p.radius);
        
        // v init légère et vers le haut
        p.vx = frand(vmin, vmax);
        p.vy = frand(vmin, vmax);

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

    // Récupérer paramètres physiques depuis les atomics
    float eps = elasticity.load(std::memory_order_relaxed);     // coefficient d'élasticité
    float mu = frictionCoeff.load(std::memory_order_relaxed);  // frottement visqueux

    // bornes raisonnables
    if (eps < 0.0f) eps = 0.0f;
    if (eps > 1.0f) eps = 1.0f;
    if (mu < 0.0f) mu = 0.0f;

    // Damping effectif par frame (base * frottement)
    float perStepDamp = damping * (1.0f - mu * dt);
    if (perStepDamp < 0.0f) perStepDamp = 0.0f;
    if (perStepDamp > 1.0f) perStepDamp = 1.0f;

    const int n = static_cast<int>(particles.size());
    if (n <= 0) return;

	// Integration  physique simple (gravite, frottement, collisions mur)
    for (auto& p : particles) {
        // gravité
        p.vy += gravityY * dt;

        // frottement global
        p.vx *= perStepDamp;
        p.vy *= perStepDamp;

        // intégration
        p.x += p.vx * dt;
        p.y += p.vy * dt;

        float r = p.radius;

        // collision horizontale
        if (p.x < r) {
            p.x = r;
            p.vx = -p.vx * eps;
        }
        else if (p.x > width - r) {
            p.x = width - r;
            p.vx = -p.vx * eps;
        }

        // collision verticale
        if (p.y < r) {
            p.y = r;
            p.vy = -p.vy * eps;
        }
        else if (p.y > height - r) {
            p.y = height - r;
            p.vy = -p.vy * eps;
        }
    }

    // Collisions inter-particules
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            Particle& a = particles[i];
            Particle& b = particles[j];

            float dx = b.x - a.x;
            float dy = b.y - a.y;
            float dist2 = dx * dx + dy * dy;

            float rSum = a.radius + b.radius;
            float rSum2 = rSum * rSum;

            if (dist2 >= rSum2 || dist2 <= 1e-6f) {
                continue; // pas de collision ou trop proche numériquement
            }

            float dist = std::sqrt(dist2);
            float nx = dx / dist;
            float ny = dy / dist;

            // Correction de recouvrement (on sépare les cercles)
            float overlap = rSum - dist;
            float half = 0.5f * overlap;
            a.x -= nx * half;
            a.y -= ny * half;
            b.x += nx * half;
            b.y += ny * half;

            // Vitesse relative le long de la normale
            float rvx = b.vx - a.vx;
            float rvy = b.vy - a.vy;
            float vn = rvx * nx + rvy * ny;

            // si les particules s'écartent déjà, pas de rebond
            if (vn > 0.0f) continue;

            // Masse = 1 pour tout le monde => formule simplifiée
            float val = -(1.0f + eps) * vn / 2.0f;
            float impulseX = val * nx;
            float impulseY = val * ny;

            a.vx -= impulseX;
            a.vy -= impulseY;
            b.vx += impulseX;
            b.vy += impulseY;
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
    if (count != lastUploadedCount) {
        cuda_particles_upload(particles.data(), count);
        lastUploadedCount = count;
    }

    // Récupération des paramètres physiques
    float eps = elasticity.load(std::memory_order_relaxed);
    float mu = frictionCoeff.load(std::memory_order_relaxed);

    // physics on GPU
    cuda_particles_step(dt, gravityY, damping, eps, mu, width, height, count);

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

            bool isPaused = paused.load(std::memory_order_relaxed);

#ifdef USE_CUDA
            bool gpuNow = useGPU.load(std::memory_order_relaxed);
            SetWindowTitle(gpuNow ? "PartiQle Studio - GPU mode"
                : "PartiQle Studio - CPU mode");
#endif

            // =====================
            // 1) PHYSIQUE
            // =====================
            if (!isPaused) {
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
            }

            // =====================
            // 2) RENDU
            // =====================
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

            // =====================
            // 3) OVERLAY "PAUSE"
            // =====================
            if (isPaused && !particles.empty()) {
                int w = curW.load();
                int h = curH.load();
                if (w <= 0) w = 800;
                if (h <= 0) h = 450;

                // léger voile sombre
                DrawRectangle(0, 0, w, h, Fade(BLACK, 0.4f));

                // symbole "||" au centre
                int size = h / 4;
                int barWidth = size / 4;
                int barHeight = size;
                int cx = w / 2;
                int cy = h / 2;

                Color pauseColor = RAYWHITE;

                // barre gauche
                DrawRectangle(cx - barWidth - barWidth / 2,
                    cy - barHeight / 2,
                    barWidth,
                    barHeight,
                    pauseColor);

                // barre droite
                DrawRectangle(cx + barWidth / 2,
                    cy - barHeight / 2,
                    barWidth,
                    barHeight,
                    pauseColor);

                const char* txt = "PAUSE";
                int fontSize = 40;
                int textWidth = MeasureText(txt, fontSize);
                DrawText(txt,
                    cx - textWidth / 2,
                    cy + barHeight / 2 + 10,
                    fontSize,
                    RAYWHITE);
            }

            EndDrawing();

            // =====================
            // 4) STATS POUR Qt
            // =====================
            if (dt > 0.0f) {
                lastFrameMs.store(dt * 1000.0f, std::memory_order_relaxed);
                lastFps.store(1.0f / dt, std::memory_order_relaxed);
            }
            lastParticleCount.store((int)particles.size(), std::memory_order_relaxed);
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


