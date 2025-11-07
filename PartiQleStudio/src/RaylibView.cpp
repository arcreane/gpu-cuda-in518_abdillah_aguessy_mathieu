#include "RaylibView.h"
#include "cuda_api.h"

// ===== Raylib =====
#include "raylib.h"

// Qt
#include <QResizeEvent>
#include <QWindow>

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

        // Optionnel : fenetre sans bordure si voulu
        // SetConfigFlags(FLAG_WINDOW_UNDECORATED);

        // 1) Creer la fenetre Raylib
        InitWindow(W, H, "Raylib embedded in Qt");
        SetTargetFPS(60);

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

            BeginDrawing();
            Color bg = { (unsigned char)clrR.load(), (unsigned char)clrG.load(), (unsigned char)clrB.load(), 255 };
            ClearBackground(bg);

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