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

    // Récupérer le widget promu pour le graphe
    fpsGraph = qobject_cast<FpsGraphWidget*>(ui.fpsGraphWidget);
    if (fpsGraph) {
        fpsGraph->setHistorySize(150);
    }

    // QFrame caché au démarrage
    if (ui.frameGraphFps) {
        ui.frameGraphFps->setVisible(false);
    }
    if (ui.actionShowGraph) {
        ui.actionShowGraph->setChecked(false);
    }

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

    // Paramètre physique  
    rlView->setElasticity(static_cast<float>(ui.spinElasticity->value()));
    rlView->setFriction(static_cast<float>(ui.spinFriction->value()));
    rlView->setGravity(static_cast<float>(ui.spinGravity->value()));
	rlView->setDamping(static_cast<float>(ui.spinDamping->value()));

    rlView->setMouseRadius(ui.spinMouseRadius->value());
    rlView->setMouseForce(static_cast<float>(ui.spinMouseForce->value()));

    if (ui.sliderVmin) {
        rlView->setVelocityMin(static_cast<float>(ui.sliderVmin->value()));
    }
    if (ui.sliderVmax) {
        rlView->setVelocityMax(static_cast<float>(ui.sliderVmax->value()));
    }

}

MainWindow::~MainWindow()
{}

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
    if (ui.sliderVmin) ui.sliderVmin->setEnabled(false);
    if (ui.sliderVmax) ui.sliderVmax->setEnabled(false);
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
    if (ui.sliderVmin) ui.sliderVmin->setEnabled(true);
    if (ui.sliderVmax) ui.sliderVmax->setEnabled(true);
    if (fpsGraph) fpsGraph->reset();
}

/* ============ spinParticles → RaylibView ============ */
void MainWindow::on_spinParticles_valueChanged(int value)
{
    if (!rlView) return;
    ui.labelParticleCount->setText(QString("Particles: %1").arg(value));
}

/* ============ spinElasticity → RaylibView ============ */
void MainWindow::on_spinElasticity_valueChanged(double value)
{
    if (!rlView) return;
    rlView->setElasticity(static_cast<float>(value));
}

/* ============ spinFriction → RaylibView ============ */
void MainWindow::on_spinFriction_valueChanged(double value)
{
    if (!rlView) return;
    rlView->setFriction(static_cast<float>(value));
}

/* ============ spinGravity, spinDamping → RaylibView ============ */
void MainWindow::on_spinGravity_valueChanged(double value)
{
    if (!rlView) return;
    rlView->setGravity(static_cast<float>(value));
}

void MainWindow::on_spinDamping_valueChanged(double value)
{
    if (!rlView) return;
    rlView->setDamping(static_cast<float>(value));
}

/* ============ Sliders Rmin, Rmax, Vmin, Vmax → RaylibView ============ */
void MainWindow::on_sliderRmin_valueChanged(int value)
{
    if (!rlView) return;

    if (value < 1) {
        ui.sliderRmin->setValue(1);
        value = 1;
    }

    float rmin = static_cast<float>(value);
    rlView->setRadiusMin(rmin);

    if (ui.sliderRmax && ui.sliderRmax->value() < value) {
        ui.sliderRmax->setValue(value);
    }

    ui.labelRmin->setText(QString("R min: %1").arg(value));
}

void MainWindow::on_sliderRmax_valueChanged(int value)
{
    if (!rlView) return;

    if (value < 1) {
        ui.sliderRmax->setValue(1);
        value = 1;
    }

    float rmax = static_cast<float>(value);
    rlView->setRadiusMax(rmax);

    if (ui.sliderRmin && ui.sliderRmin->value() > value) {
        ui.sliderRmin->setValue(value);
    }

    ui.labelRmax->setText(QString("R max: %1").arg(value));
}

void MainWindow::on_sliderVmin_valueChanged(int value)
{
    if (!rlView) return;
    rlView->setVelocityMin(static_cast<float>(value));
    
    // afficher la valeur dans un label
    ui.labelVmin->setText(QString("V min: %1").arg(value));
}

void MainWindow::on_sliderVmax_valueChanged(int value)
{
    if (!rlView) return;
    rlView->setVelocityMax(static_cast<float>(value));

    // afficher la valeur dans un label
    ui.labelVmax->setText(QString("V max: %1").arg(value));

}

void MainWindow::on_spinMouseRadius_valueChanged(int value)
{
    if (!rlView) return;
    rlView->setMouseRadius(value);
}

void MainWindow::on_spinMouseForce_valueChanged(double value)
{
    if (!rlView) return;
    rlView->setMouseForce(static_cast<float>(value));
}

/* ============ Debug Overlays ============ */
void MainWindow::on_actionShowAllInfos_toggled(bool checked)
{
    if (!rlView) return;

    // Bloquer les signaux pour éviter les boucles infinies
    ui.actionShowMouseInfo->blockSignals(true);
    ui.actionShowEngineInfo->blockSignals(true);
    ui.actionShowPerfInfo->blockSignals(true);
    ui.actionShowBoxsimInfo->blockSignals(true);

    // Synchroniser toutes les actions
    ui.actionShowMouseInfo->setChecked(checked);
    ui.actionShowEngineInfo->setChecked(checked);
    ui.actionShowPerfInfo->setChecked(checked);
    ui.actionShowBoxsimInfo->setChecked(checked);

    // Débloquer les signaux
    ui.actionShowMouseInfo->blockSignals(false);
    ui.actionShowEngineInfo->blockSignals(false);
    ui.actionShowPerfInfo->blockSignals(false);
    ui.actionShowBoxsimInfo->blockSignals(false);

    // Appliquer les changements à RaylibView
    rlView->setShowMouseInfo(checked);
    rlView->setShowEngineInfo(checked);
    rlView->setShowPerfInfo(checked);
    rlView->setShowBoxsimInfo(checked);
}

void MainWindow::on_actionShowMouseInfo_toggled(bool checked)
{
    if (!rlView) return;
    rlView->setShowMouseInfo(checked);
    if (!checked && ui.actionShowAllInfos->isChecked()) {
        ui.actionShowAllInfos->blockSignals(true);
        ui.actionShowAllInfos->setChecked(false);
        ui.actionShowAllInfos->blockSignals(false);
    }
}

void MainWindow::on_actionShowEngineInfo_toggled(bool checked)
{
    if (!rlView) return;
    rlView->setShowEngineInfo(checked);
    if (!checked && ui.actionShowAllInfos->isChecked()) {
        ui.actionShowAllInfos->blockSignals(true);
        ui.actionShowAllInfos->setChecked(false);
        ui.actionShowAllInfos->blockSignals(false);
    }
}

void MainWindow::on_actionShowPerfInfo_toggled(bool checked)
{
    if (!rlView) return;
    rlView->setShowPerfInfo(checked);
    if (!checked && ui.actionShowAllInfos->isChecked()) {
        ui.actionShowAllInfos->blockSignals(true);
        ui.actionShowAllInfos->setChecked(false);
        ui.actionShowAllInfos->blockSignals(false);
    }
}

void MainWindow::on_actionShowBoxsimInfo_toggled(bool checked)
{
	if (!rlView) return;
	rlView->setShowBoxsimInfo(checked);
    if (!checked && ui.actionShowAllInfos->isChecked()) {
        ui.actionShowAllInfos->blockSignals(true);
        ui.actionShowAllInfos->setChecked(false);
        ui.actionShowAllInfos->blockSignals(false);
    }
}

void MainWindow::on_actionShowGraph_toggled(bool checked)
{
    if (ui.frameGraphFps){
        ui.frameGraphFps->setVisible(checked);
    }
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

    // --- Update du graphe FPS/FrameTime ---
    if (fpsGraph && ui.frameGraphFps && ui.frameGraphFps->isVisible()) {
        if (!rlView->isPaused()) {
            fpsGraph->pushSample(fps, ms);
        }
    }
}

void MainWindow::on_actionPresetTerre_triggered()
{
    // Physique “Terre”
    if (ui.sliderVmin) ui.sliderVmin->setValue(5);
    if (ui.sliderVmax) ui.sliderVmax->setValue(30);
    ui.spinElasticity->setValue(0.85);
    ui.spinFriction->setValue(0.3);
	ui.spinDamping->setValue(0.995);
    ui.spinGravity->setValue(120);
}

void MainWindow::on_actionPresetMars_triggered()
{
    // Physique “Mars”
    if (ui.sliderVmin) ui.sliderVmin->setValue(8);
    if (ui.sliderVmax) ui.sliderVmax->setValue(40);
    ui.spinElasticity->setValue(0.8);
    ui.spinFriction->setValue(0.1);
    ui.spinDamping->setValue(0.995);
    ui.spinGravity->setValue(60);
}

void MainWindow::on_actionPresetVideSpatial_triggered()
{
    // Physique “Vide spatial”
    if (ui.sliderVmin) ui.sliderVmin->setValue(0);
    if (ui.sliderVmax) ui.sliderVmax->setValue(80);
    ui.spinElasticity->setValue(0.95);
    ui.spinFriction->setValue(0.0);
    ui.spinDamping->setValue(0.999);
    ui.spinGravity->setValue(0);
}

void MainWindow::on_actionReset_Param_Physique_triggered()
{
    // Paramètres par défaut
    if (ui.sliderVmin) ui.sliderVmin->setValue(20);
    if (ui.sliderVmax) ui.sliderVmax->setValue(20);
    ui.spinElasticity->setValue(0.80);
    ui.spinFriction->setValue(0.20);
    ui.spinDamping->setValue(0.999);
    ui.spinGravity->setValue(0);
}
