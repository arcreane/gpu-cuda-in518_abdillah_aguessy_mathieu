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

void RaylibView::setMouseRadius(int radius) {
    mouseRadius.store(radius, std::memory_order_relaxed);
}

void RaylibView::setMouseForce(float force) {
    mouseForceScale.store(force, std::memory_order_relaxed);
}

void RaylibView::setShowMouseInfo(bool v) {
    showMouseInfo.store(v, std::memory_order_relaxed);
}

void RaylibView::setShowEngineInfo(bool v) {
    showEngineInfo.store(v, std::memory_order_relaxed);
}

void RaylibView::setShowPerfInfo(bool v) {
    showPerfInfo.store(v, std::memory_order_relaxed);
}

void RaylibView::setShowBoxsimInfo(bool v) {
    showBoxsimInfo.store(v, std::memory_order_relaxed);
}

bool RaylibView::isShowMouseInfo() const {
    return showMouseInfo.load(std::memory_order_relaxed);
}

bool RaylibView::isShowEngineInfo() const {
    return showEngineInfo.load(std::memory_order_relaxed);
}

bool RaylibView::isShowPerfInfo() const {
    return showPerfInfo.load(std::memory_order_relaxed);
}

bool RaylibView::isShowBoxsimInfo() const {
    return showBoxsimInfo.load(std::memory_order_relaxed);
}

void RaylibView::applyMouseForceCPU(float mouseX, float mouseY, float velX, float velY, int mode) {
    float radius = static_cast<float>(mouseRadius.load(std::memory_order_relaxed));
    float forceScale = mouseForceScale.load(std::memory_order_relaxed);
    const float radiusSq = radius * radius;

    for (auto& p : particles) {
        float dx = p.x - mouseX;
        float dy = p.y - mouseY;
        float distSq = dx * dx + dy * dy;

        if (distSq < radiusSq && distSq > 1e-6f) {
            float dist = std::sqrt(distSq);
            float nx = dx / dist; // Normale depuis souris vers particule
            float ny = dy / dist;

            float influence = 1.0f - (dist / radius); // Décroissance linéaire

            switch (mode) {
            case 0: // Survol : Pousse dans la direction du mouvement
            {
                p.vx += velX * forceScale * influence;
                p.vy += velY * forceScale * influence;
                break;
            }

            case 1: // Clic gauche : Attire vers la souris
            {
                float attractForce = forceScale * 200.0f * influence;
                p.vx -= nx * attractForce;
                p.vy -= ny * attractForce;
                break;
            }

            case 2: // Clic droit : Explosion (repousse)
            {
                float explosionForce = forceScale * 300.0f * influence;
                p.vx += nx * explosionForce;
                p.vy += ny * explosionForce;
                break;
            }
            }
        }
    }
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

        int actualW = GetScreenWidth();
        int actualH = GetScreenHeight();
        curW.store(actualW);
        curH.store(actualH);

        // 2) Recuperer le handle natif et le transmettre a Qt
        void* native = GetWindowHandle();
        p->set_value(native);

        using clock = std::chrono::steady_clock;
        auto lastTime = clock::now();

		// variables souris
        float prevMouseX = 0.0f;
        float prevMouseY = 0.0f;

        int prevW = actualW;
        int prevH = actualH;

        // 3) rendu raylib
        while (running.load() && !WindowShouldClose()) {
            actualW = GetScreenWidth();
            actualH = GetScreenHeight();
            
            bool dimensionsChanged = (actualW != prevW || actualH != prevH);
            if (dimensionsChanged) {
                curW.store(actualW);
                curH.store(actualH);
                prevW = actualW;
                prevH = actualH;

                // Repositionner les particules qui sortent des nouveaux murs
                for (auto& part : particles) {
                    float r = part.radius;

                    // Clamper dans les nouvelles limites
                    if (part.x < r) part.x = r;
                    else if (part.x > actualW - r) part.x = actualW - r;

                    if (part.y < r) part.y = r;
                    else if (part.y > actualH - r) part.y = actualH - r;
                }
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
            // Gestion de la souris (3 modes)
            Vector2 mousePos = GetMousePosition();
            float mouseX = mousePos.x;
            float mouseY = mousePos.y;

            bool isLeftMouseDown = IsMouseButtonDown(MOUSE_BUTTON_LEFT);
            bool isRightMouseDown = IsMouseButtonDown(MOUSE_BUTTON_RIGHT);

            // Calculer la vitesse du curseur (en pixels/sec)
            float mouseVelX = (mouseX - prevMouseX) / dt;
            float mouseVelY = (mouseY - prevMouseY) / dt;

            prevMouseX = mouseX;
            prevMouseY = mouseY;

            int mouseMode = -1; // -1 = pas d'interaction
            if (isRightMouseDown) {
                mouseMode = 2; // Explosion
            }
            else if (isLeftMouseDown) {
                mouseMode = 1; // Attraction
            }
            else {
                // Survol uniquement si la souris bouge suffisamment
                float speed = std::sqrt(mouseVelX * mouseVelX + mouseVelY * mouseVelY);
                if (speed > 10.0f) { // Seuil de vitesse
                    mouseMode = 0; // Pousse
                }
            }

            // =====================
            // 1) PHYSIQUE
            // =====================
            if (!isPaused) {
#ifdef USE_CUDA
                if (useGPU.load(std::memory_order_relaxed)) {
                    stepParticlesGPU(dt);

                    // Appliquer la force de la souris (GPU)
                    if (mouseMode >= 0) {
                        cuda_particles_download(particles.data(), particles.size());
                        applyMouseForceCPU(mouseX, mouseY, mouseVelX, mouseVelY, mouseMode);
                        cuda_particles_upload(particles.data(), particles.size());
                    }
                }
                else {
                    stepParticlesCPU(dt);

                    // Appliquer la force de la souris (CPU)
                    if (mouseMode >= 0) {
                        applyMouseForceCPU(mouseX, mouseY, mouseVelX, mouseVelY, mouseMode);
                    }
                }
#else
                stepParticlesCPU(dt);

                // Appliquer la force de la souris (CPU uniquement)
                if (mouseMode >= 0) {
                    applyMouseForceCPU(mouseX, mouseY, mouseVelX, mouseVelY, mouseMode);
                }
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

			// Bordure et taille simulation
            if (setShowBoxsimInfo.load(std::memory_order_relaxed)) {
                DrawRectangleLines(0, 0, actualW, actualH, Fade(RED, 0.5f));
                DrawText(TextFormat("Sim: %dx%d", actualW, actualH), actualW - 150, 10, 20, RED);
            }

			// Dessin des particules
            for (const auto& part : particles) {
                Color c = { part.r, part.g, part.b, part.a };
                DrawCircleV(Vector2{ part.x, part.y }, part.radius, c);
            }

            //Afficher le rayon d'influence de la souris
            if (mouseMode >= 0) {
                int radius = mouseRadius.load(std::memory_order_relaxed);
                Color circleColor;

                switch (mouseMode) {
                case 0: // Survol : Jaune
                    circleColor = Fade(YELLOW, 0.3f);
                    break;
                case 1: // Attraction : Vert
                    circleColor = Fade(GREEN, 0.4f);
                    break;
                case 2: // Explosion : Rouge
                    circleColor = Fade(RED, 0.5f);
                    break;
                }

                DrawCircleLines((int)mouseX, (int)mouseY, radius, circleColor);
                DrawCircleV(Vector2{ mouseX, mouseY }, 5.0f, circleColor);
            }

            // =====================
            // INFOS MOTEUR (DEBUG)
            // =====================
            if (showEngineInfo.load(std::memory_order_relaxed)) {
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
            }

            // =====================
            // INFOS SOURIS (DEBUG)
            // =====================
            if (showMouseInfo.load(std::memory_order_relaxed)) {
                // Afficher position souris
                DrawText(TextFormat("Mouse: (%.0f, %.0f)", mouseX, mouseY), 10, 70, 20, BLUE);
                
                // Afficher mode souris
                if (mouseMode >= 0) {
                    const char* modeText = "";
                    switch (mouseMode) {
                    case 0: modeText = "POUSSE"; break;
                    case 1: modeText = "ATTIRE"; break;
                    case 2: modeText = "EXPLOSION"; break;
                    }
                    DrawText(modeText, 10, 95, 20, BLUE);
                }
            }

            // =====================
            // INFOS PERFORMANCE (DEBUG)
            // =====================
            if (showPerfInfo.load(std::memory_order_relaxed)) {
                float currentFps = lastFps.load(std::memory_order_relaxed);
                float currentFrameMs = lastFrameMs.load(std::memory_order_relaxed);
                DrawText(TextFormat("FPS: %.1f (%.2f ms)", currentFps, currentFrameMs),
                    10, 120, 20, ORANGE);
            }

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





