#pragma once

#include <QWidget>
#include <QVector>

class FpsGraphWidget : public QWidget
{
    Q_OBJECT
public:
    explicit FpsGraphWidget(QWidget* parent = nullptr);

    void setHistorySize(int n);
    void reset();
    void pushSample(float fps, float frameMs);

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    int m_historySize = 150;          // ~150 * 200 ms ≃ 30 s d’historique
    QVector<float> m_fpsHistory;      // FPS
    QVector<float> m_msHistory;       // FrameTime (ms)

    float maxInVector(const QVector<float>& v) const;
};

