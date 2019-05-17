#include "histogram.h"
#include "math.h"
#include "stdio.h"
#include <iostream>
#include "Settings.h"

Histogram::Histogram(std::vector<float> luminances)
{
    //m_histogram.reserve(m_size);
    for (int i = 0; i < m_size; i++) {
        m_histogram.push_back(0);
    }

    // Take the log of all the luminances and ignore negative values
    for (int i = 0; i < luminances.size(); i++) {
        //float logVal = log(luminances[i]);

        m_luminances.push_back(luminances[i]);
    }

    float max = -1;
    for (int i  = 0; i < m_luminances.size(); i++) {
        if (m_luminances[i] > max) {
            max = m_luminances[i];
        }
    }

    m_binSize = max / (float)m_size;
    createHistogram();

}

void Histogram::createHistogram()
{

    for (int i = 0; i < m_luminances.size(); i++) {
        float lum = m_luminances[i];
        if (lum <= 0) {
           // std::cout << "continuing" << std::endl;
            continue;
        }
        int bin = (int) ((lum - fmod(lum, m_binSize)) / m_binSize);
        //std::cout << "Incrementing hist" << std::endl;
        m_histogram[bin]++;
    }

}

float Histogram::findSlope(int bin)
{
    int x1 = bin;
    int x2 = bin + 1;
    int y1 = m_histogram[x1];
    int y2 = m_histogram[x2];

    float slope = ((float)(y2 - y1)) / ((float)(x2 - x1));
    return slope;
}

float Histogram::findLowestSlope()
{
    float minSlope = 100000000.f;
    int minIndex = 0;


    float histogramAccumulator = 0.0f;
//    for (int i = 0; i < m_histogram.size() - 1; i++) {
    for (int i = 0; i < m_histogram.size(); i++) {
        histogramAccumulator += float(m_histogram[i]);
//        float slope = findSlope(i);
//        std::cout << i << " " << m_histogram[i] << std::endl;
//        if (slope < minSlope) {
//            minSlope = slope;
//            minIndex = i;
//        }
    }
    int cutoffIndex = m_histogram.size() - 1;
    float cutoffAccumulator = 0.0f;
    for (int i = m_histogram.size() - 1; i >= 0; i--) {
        cutoffAccumulator += float(m_histogram[i]);

        if(cutoffAccumulator/histogramAccumulator > (settings.ht / 100)){

            cutoffIndex = i;
            break;
        }
    }
    minIndex = cutoffIndex;
    std::cout << "Slope index: " << minIndex << std::endl;

    // Returning beginning of highlight vals
    return ((float)minIndex) * m_binSize ;//+ 0.5;

}

std::vector<int> Histogram::findHighlights()
{
    float lowestLog = findLowestSlope();
    std::cout << "Lowest log: " << lowestLog << std::endl;
    std::vector<int> highlights;
    for (int i = 0; i < m_luminances.size(); i++) {

        float lum = m_luminances[i];
        if (lum > lowestLog) {
            highlights.push_back(i);
        }
    }

    std::cout << "Found highlights" << std::endl;
    return highlights;
}

std::vector<float> Histogram::getHighlightsMaxAndMin()
{
    float lowestLog = findLowestSlope();
    std::cout << "Lowest log: " << lowestLog << std::endl;
    std::vector<float> highlights;
    highlights.push_back(lowestLog);
    float max = -10000000.f;
    for (int i = 0; i < m_luminances.size(); i++) {

        float lum = m_luminances[i];
        if (lum > max) {
            max = lum;
        }
    }
    highlights.push_back(max);

    std::cout << "Found highlights" << std::endl;
    return highlights;
}

float Histogram::findPeakHistogramValue()
{
    int max = -1;
    int max_i = -1;
    for (int i = 0; i < m_histogram.size(); i++) {
        if (m_histogram[i] > max) {
            max = m_histogram[i];
            max_i = i;
        }
    }
    float lower_bound = max_i * m_binSize;
    float upper_bound = lower_bound + m_binSize;
    return (lower_bound + upper_bound) / 2.f;
}
