#ifndef SHAPEESTIMATION_H
#define SHAPEESTIMATION_H
#include "imagereader.h"
#include "bilateralfilter.h"
#include "Eigen/Dense"

class ShapeEstimation
{
public:
    ShapeEstimation();

    void sigmoidalCompression(std::vector<float> &pixelLuminances, float sigma);
    std::vector<float> computePixelLuminance(ImageReader imageIn, ImageReader mask, float &sigma);
    void estimateShape(ImageReader imageIn, ImageReader mask, std::vector<float>& depthMap, std::vector<Eigen::Vector3f>& normalMap, std::vector<float> &gradientX, std::vector<float> &gradientY);

    float gradientReshape(float x);
    float gradientReshapeRecursive(float in, int itr);
    std::vector<Eigen::Vector3f> gradientField(ImageReader imageIn, std::vector<float> &pixelLuminances,
                                               std::vector<float> &gradientX, std::vector<float> &gradientY);
    void sigmoidalInversion(std::vector<float> &pixelLuminances, float sigma);

    void cropMask(ImageReader mask, std::vector<float> &pixelLuminances);

    std::vector<float> getLuminances();
    float m_curvature = 1.0f;
    float m_bilateralSmoothing = 0.0f;
private:
    std::vector<float> m_luminances;

};

#endif // SHAPEESTIMATION_H
