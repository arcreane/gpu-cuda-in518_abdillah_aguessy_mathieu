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

#ifdef USE_CUDA
    // Raccourci clavier "G" pour basculer CPU <-> GPU
    auto* toggleGpuShortcut = new QShortcut(QKeySequence(Qt::Key_G), this);
    connect(toggleGpuShortcut, &QShortcut::activated, this, [this]() {
        bool now = !rlView->isUsingGPU();
        rlView->setUseGPU(now);
        });
#endif
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

