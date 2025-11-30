#include "MainWindow.h"
#include "RaylibView.h"
#include "cuda_api.h"
#include <QVBoxLayout>
#include <QMessageBox>
#include <QShortcut>
#include <QDebug>
#include <vector>

MainWindow::MainWindow(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);

	rlView = new RaylibView(this); //widget pour Raylib
	rlView->setMinimumSize(800, 450);

    if (ui.raylibPlaceholder->layout()) {
        ui.raylibPlaceholder->layout()->addWidget(rlView);
    }
    else {
        auto* lay = new QVBoxLayout(ui.raylibPlaceholder);
        lay->setContentsMargins(0, 0, 0, 0);
        lay->addWidget(rlView);
    }

	// Mode CPU par défaut
    ui.radioCPU->setChecked(true);
    ui.radioGPU->setChecked(false);
    rlView->setUseGPU(false);

	// Timer stats
    statsTimer = new QTimer(this);
    connect(statsTimer, &QTimer::timeout,
        this, &MainWindow::updateStats);
    statsTimer->start(200); // tous les 200 ms

	// Etat initial des boutons de pause
    ui.buttonPause->setEnabled(false);
    ui.buttonPause->setText("Pause");

}

MainWindow::~MainWindow()
{}

void MainWindow::on_btnRunCuda_clicked()
{
#ifdef USE_CUDA
    // ==========
    // VERSION AVEC CUDA
    // ==========
    const int grid = 3;
    const int block = 4;
    const int size = grid * block;

    std::vector<int> vBlockDim(size), vThreadIdx(size), vBlockIdx(size), vGlobalIdx(size);

    // Appel CUDA cf kernel.cu
    cuda_demo_dump(grid, block,
        vBlockDim.data(), vThreadIdx.data(),
        vBlockIdx.data(), vGlobalIdx.data());

    auto toString = [](const std::vector<int>& v, int n = 8) {
        QStringList parts;
        const int k = std::min<int>(n, (int)v.size());
        for (int i = 0; i < k; ++i) parts << QString::number(v[i]);
        if ((int)v.size() > k) parts << "...";
        return parts.join(", ");
        };

    QString msg;
    msg += "blockDim.x : [" + toString(vBlockDim) + "]\n";
    msg += "threadIdx  : [" + toString(vThreadIdx) + "]\n";
    msg += "blockIdx   : [" + toString(vBlockIdx) + "]\n";
    msg += "globalIdx  : [" + toString(vGlobalIdx) + "]";

    qDebug().noquote() << msg;
    QMessageBox::information(this, "CUDA dump", msg);

#else
    // ==========
    // VERSION SANS CUDA
    // ==========
    QMessageBox::warning(
        this,
        "CUDA non disponible",
        "Votre machine ne possède pas CUDA.\n"
        "Le calcul GPU n'a pas été exécuté."
    );
#endif
}

/* ============ Moteur CPU / GPU ============ */
void MainWindow::on_radioCPU_toggled(bool checked)
{
    if (!checked || !rlView) return;
    rlView->setUseGPU(false);
    ui.labelMode->setText("Mode: CPU");
}

void MainWindow::on_radioGPU_toggled(bool checked)
{
    if (!checked || !rlView) return;
#ifdef USE_CUDA
    rlView->setUseGPU(true);
    ui.labelMode->setText("Mode: GPU");
#else
    // Si CUDA pas compilé, on empêche le switch
    ui.radioCPU->setChecked(true);
    ui.radioGPU->setChecked(false);
    QMessageBox::warning(this, "CUDA non disponible",
        "L'exécutable actuel ne supporte pas CUDA.\n"
        "Le mode GPU n'est pas disponible sur cette machine.");
#endif
}

/* ============ Run / Pause / Reset ============ */
void MainWindow::on_buttonStart_clicked()
{
    if (!rlView) return;
    int count = ui.spinParticles->value();
    rlView->setParticleCount(count);

    rlView->setPaused(false);
    
    ui.labelParticleCount->setText(QString("Particles: %1").arg(count));
    ui.buttonPause->setEnabled(true);
    ui.buttonPause->setText("Pause");

	ui.spinParticles->setEnabled(false); // désactiver le spinbox pendant la simulation
	ui.sliderRmax->setEnabled(false);
	ui.sliderRmin->setEnabled(false);
}

void MainWindow::on_buttonPause_clicked()
{
    if (!rlView) return;
    bool currentlyPaused = rlView->isPaused();
    bool newPaused = !currentlyPaused;
    rlView->setPaused(newPaused);

    if (newPaused) {
        ui.buttonPause->setText("Reprendre");
    }
    else {
        ui.buttonPause->setText("Pause");
    }
}

void MainWindow::on_buttonReset_clicked()
{
    if (!rlView) return;
    rlView->resetSimulation();
	rlView->setPaused(true); // mettre en pause après reset

    ui.buttonPause->setEnabled(false);
    ui.buttonPause->setText("Pause");

    ui.labelParticleCount->setText("Particles: 0");

	ui.spinParticles->setEnabled(true); // réactiver le spinbox après reset
    ui.sliderRmax->setEnabled(true);
    ui.sliderRmin->setEnabled(true);
}

/* ============ spinParticles → RaylibView ============ */
void MainWindow::on_spinParticles_valueChanged(int value)
{
    if (!rlView) return;
    ui.labelParticleCount->setText(QString("Particles: %1").arg(value));
}

/* ============ Mise à jour Stats ============ */
void MainWindow::updateStats()
{
    if (!rlView) return;

    float fps = rlView->fps();
    float ms = rlView->frameTimeMs();
    int count = rlView->particleCount();
    bool gpu = rlView->isUsingGPU();

    ui.labelFPS->setText(QString("FPS: %1").arg(fps, 0, 'f', 1));
    ui.labelFrameTime->setText(QString("Frame time: %1 ms").arg(ms, 0, 'f', 2));
    ui.labelParticleCount->setText(QString("Particles: %1").arg(count));
    ui.labelMode->setText(QString("Mode: %1").arg(gpu ? "GPU" : "CPU"));
}

