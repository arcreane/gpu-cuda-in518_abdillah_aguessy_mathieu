#include "RaylibView.h"
#include "cuda_api.h"

// ===== Raylib =====
#include "raylib.h"

// Qt
#include <QResizeEvent>
#include <QWindow>
#include <cmath>


static float frand(float minVal, float maxVal) {
    int r = GetRandomValue(0, 10000);
    float t = r / 10000.0f;
    return minVal + (maxVal - minVal) * t;
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
    clrR = r; clrG = g; clrB = b;
}

void RaylibView::resizeEvent(QResizeEvent* ev) {
    QWidget::resizeEvent(ev);
    reqW = ev->size().width();
    reqH = ev->size().height();
}

/* PARICULES - Methods */
void RaylibView::initParticles(int count, int width, int height) {
    particles_.clear();
    particles_.reserve(count);

    for (int i = 0; i < count; ++i) {
        CpuParticle p;
        p.radius = frand(3.0f, 6.0f);

        p.x = frand(p.radius, width - p.radius);
        p.y = frand(p.radius, height - p.radius);

        // v init légère et vers le haut
        p.vx = frand(-50.0f, 50.0f);
        p.vy = frand(-80.0f, -20.0f);

        p.r = (unsigned char)(100 + (i * 13) % 155);
        p.g = (unsigned char)(100 + (i * 29) % 155);
        p.b = (unsigned char)(180 + (i * 7) % 75);
        p.a = 255;

        particles_.push_back(p);
    }
}

void RaylibView::updateParticlesCPU(float dt, int width, int height) {
    for (auto& p : particles_) {
        // gravite
        p.vy += gravity_ * dt;

        // frottement global
        p.vx *= damping_;
        p.vy *= damping_;

        // integration
        p.x += p.vx * dt;
        p.y += p.vy * dt;

        float r = p.radius;

        // collision horizontale
        if (p.x < r) {
            p.x = r;
            p.vx = -p.vx * 0.8f; // rebond amorti
        }
        else if (p.x > width - r) {
            p.x = width - r;
            p.vx = -p.vx * 0.8f;
        }

        // collision verticale
        if (p.y < r) {
            p.y = r;
            p.vy = -p.vy * 0.8f;
        }
        else if (p.y > height - r) {
            p.y = height - r;
            p.vy = -p.vy * 0.8f;
        }
    }
}


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

		// Initialiser les particules
        initParticles(600, W, H);

        // 2) Recuperer le handle natif et le transmettre a Qt
        void* native = GetWindowHandle();
        p->set_value(native);

        // 3) Boucle de rendu Raylib
        while (running && !WindowShouldClose()) {
            // Suivre le resize demande par Qt
            int rw = reqW.load(), rh = reqH.load();
            if ((rw != curW) || (rh != curH)) {
                if (rw > 0 && rh > 0) {
                    SetWindowSize(rw, rh);
                    curW = rw; curH = rh;
                }
            }

            int cw = curW.load();
            int ch = curH.load();

			int simW = (cw > 0 ? cw : W);
            int simH = (ch > 0 ? ch : H);
			float dt = GetFrameTime();
			updateParticlesCPU(dt, simW, simH);

            BeginDrawing();
            Color bg = { (unsigned char)clrR.load(), (unsigned char)clrG.load(), (unsigned char)clrB.load(), 255 };
            ClearBackground(bg);

			// Dessiner les particules
            for (const auto& p : particles_) {
                Vector2 pos{ p.x, p.y };
                Color   col{ p.r, p.g, p.b, p.a };
                DrawCircleV(pos, p.radius, col);

            }
            // Exemple minimal de drawing (remarque : DrawText utilise la police par defaut)
            DrawText("Raylib inside Qt", 10, 10, 20, MAROON);

            EndDrawing();
        }

        // Fermer proprement la fenetre Raylib
        CloseWindow();
        running = false;
        });

    // Recuperer le handle natif et l'embedder dans l'UI Qt (sur le thread Qt)
    void* native = f.get();
    embedHandleToQt(native);
}

void RaylibView::stopRaylibThread() {
    if (!running) return;
    running = false;
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
    }
    else {
        auto* l = new QVBoxLayout(this);
        l->setContentsMargins(0, 0, 0, 0);
        l->addWidget(containerWidget);
        setLayout(l);
    }
}