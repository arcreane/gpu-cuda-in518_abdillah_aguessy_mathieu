#include "FpsGraphWidget.h"
#include <QPainter>
#include <QPaintEvent>
#include <algorithm>
#include <cmath>

FpsGraphWidget::FpsGraphWidget(QWidget* parent)
    : QWidget(parent)
{
    setMinimumHeight(100);
}

void FpsGraphWidget::setHistorySize(int n)
{
    if (n <= 0) return;
    m_historySize = n;
    if (m_fpsHistory.size() > m_historySize) {
        m_fpsHistory = m_fpsHistory.mid(m_fpsHistory.size() - m_historySize);
    }
    if (m_msHistory.size() > m_historySize) {
        m_msHistory = m_msHistory.mid(m_msHistory.size() - m_historySize);
    }
    update();
}

void FpsGraphWidget::reset()
{
    m_fpsHistory.clear();
    m_msHistory.clear();
    update();
}

void FpsGraphWidget::pushSample(float fps, float frameMs)
{
    if (m_historySize <= 0) return;

    if (m_fpsHistory.size() >= m_historySize) {
        m_fpsHistory.pop_front();
    }
    if (m_msHistory.size() >= m_historySize) {
        m_msHistory.pop_front();
    }

    m_fpsHistory.push_back(fps);
    m_msHistory.push_back(frameMs);

    update();   // déclenche un repaint
}

float FpsGraphWidget::maxInVector(const QVector<float>& v) const
{
    if (v.isEmpty()) return 0.0f;
    float m = 0.0f;
    for (float x : v) {
        if (x > m) m = x;
    }
    return m;
}

void FpsGraphWidget::paintEvent(QPaintEvent* event)
{
    Q_UNUSED(event);

    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, true);

    const int w = width();
    const int h = height();
    if (w <= 10 || h <= 10) {
        return;
    }

    // Fond
    p.fillRect(rect(), QColor(18, 18, 18));

    const int marginLeft   = 40;
    const int marginRight  = 10;
    const int marginTop    = 10;
    const int marginBottom = 20;

    QRect drawRect(marginLeft,
                   marginTop,
                   w - marginLeft - marginRight,
                   h - marginTop - marginBottom);

    if (drawRect.width() <= 0 || drawRect.height() <= 0) {
        return;
    }

    // Grille légère
    p.setPen(QColor(60, 60, 60));
    int gridLines = 4;
    for (int i = 0; i <= gridLines; ++i) {
        int y = drawRect.top() + i * drawRect.height() / gridLines;
        p.drawLine(drawRect.left(), y, drawRect.right(), y);
    }

    // Si pas de données, rien à tracer
    int n = m_fpsHistory.size();
    if (n < 2) {
        // petit texte "no data"
        p.setPen(QColor(160, 160, 160));
        p.drawText(drawRect, Qt::AlignCenter, "No samples yet");
        return;
    }

    // Échelle FPS
    float maxFps = maxInVector(m_fpsHistory);
    if (maxFps < 10.0f) maxFps = 10.0f;    // minimum 10 FPS pour l’échelle

    // Échelle FrameTime (ms)
    float maxMs = maxInVector(m_msHistory);
    if (maxMs < 5.0f) maxMs = 5.0f;

    // Construire points (ancien → gauche, récent → droite)
    auto buildPolyline = [&](const QVector<float>& data, bool isMs) {
        QVector<QPointF> poly;
        poly.reserve(data.size());

        const int nb = data.size();
        for (int i = 0; i < nb; ++i) {
            float xRatio = (nb <= 1) ? 0.0f : (float)i / (float)(nb - 1);
            float value = data[i];

            float norm = 0.0f;
            if (isMs) {
                norm = value / maxMs;
            } else {
                norm = value / maxFps;
            }
            if (norm > 1.0f) norm = 1.0f;
            if (norm < 0.0f) norm = 0.0f;

            float x = drawRect.left() + xRatio * drawRect.width();
            float y = drawRect.bottom() - norm * drawRect.height();
            poly.push_back(QPointF(x, y));
        }
        return poly;
    };

    auto lineFps = buildPolyline(m_fpsHistory, false);
    auto lineMs  = buildPolyline(m_msHistory, true);

    // Tracer FPS (vert)
    p.setPen(QPen(QColor(0, 220, 0), 2));
    p.drawPolyline(lineFps.constData(), lineFps.size());

    // Tracer FrameTime (orange)
    p.setPen(QPen(QColor(255, 160, 0), 2));
    p.drawPolyline(lineMs.constData(), lineMs.size());

    // Légende
    p.setPen(Qt::white);
    p.drawText(marginLeft, marginTop - 2, QString("FPS (max ~ %1)").arg(maxFps, 0, 'f', 1));
    p.drawText(marginLeft, h - 4, QString("Frame time (max ~ %1 ms)").arg(maxMs, 0, 'f', 1));

    // Mini légende couleur
    int legendY = marginTop + 15;
    p.setPen(Qt::NoPen);
    p.setBrush(QColor(0, 220, 0));
    p.drawRect(marginLeft, legendY, 10, 3);
    p.setPen(Qt::white);
    p.drawText(marginLeft + 15, legendY + 5, "FPS");

    legendY += 15;
    p.setPen(Qt::NoPen);
    p.setBrush(QColor(255, 160, 0));
    p.drawRect(marginLeft, legendY, 10, 3);
    p.setPen(Qt::white);
    p.drawText(marginLeft + 15, legendY + 5, "Frame time (ms)");
}
