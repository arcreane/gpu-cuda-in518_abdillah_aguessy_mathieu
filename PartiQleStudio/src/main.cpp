#include <QApplication>
#include "MainWindow.h"

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);

    MainWindow w;
    w.resize(1000, 700);
    w.setWindowTitle("PartiQle Studio");
	w.show();   

    return app.exec();
}
