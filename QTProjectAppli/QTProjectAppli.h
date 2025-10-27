#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_QTProjectAppli.h"

class QTProjectAppli : public QMainWindow
{
    Q_OBJECT

public:
    QTProjectAppli(QWidget *parent = nullptr);
    ~QTProjectAppli();

private:
    Ui::QTProjectAppliClass ui;
};

