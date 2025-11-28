#pragma once

#include <QMainWindow>
#include <QTimer>
#include "ui_MainWindow.h"

class RaylibView;

class MainWindow : public QMainWindow
{
	Q_OBJECT

public:
	explicit MainWindow(QWidget *parent = nullptr);
	~MainWindow();


private slots:
	void on_btnRunCuda_clicked();

	void on_radioCPU_toggled(bool checked);
	void on_radioGPU_toggled(bool checked);

	void on_buttonStart_clicked();
	void on_buttonPause_clicked();
	void on_buttonReset_clicked();

	void on_spinParticles_valueChanged(int value);

	void updateStats();

private:
	Ui::MainWindowClass ui;
	RaylibView* rlView = nullptr;

	QTimer* statsTimer = nullptr;
};

