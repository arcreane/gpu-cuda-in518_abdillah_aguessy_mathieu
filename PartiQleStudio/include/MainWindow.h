#pragma once

#include <QMainWindow>
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

private:
	Ui::MainWindowClass ui;
	RaylibView* rlView = nullptr;
};

