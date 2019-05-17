#ifndef BRDFREPLACEMENT_H
#define BRDFREPLACEMENT_H

#include "math.h"
#include "assert.h"
#include <QImage>
#include "Eigen/Dense"
#include <vector>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <map>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseQR>
#include <Eigen/Sparse>

using namespace Eigen;
struct SHSample{
    Vector3f sph;
    Vector3f vec;
    std::vector<float> coeff;
};

class BrdfReplacement
{
public:
    BrdfReplacement();
    typedef Eigen::Triplet<float> Triple;

    std::vector<Eigen::Vector3f> sample(std::vector<Eigen::Vector3f> inpainting, std::vector<Eigen::Vector3f> mask, std::vector<Eigen::Vector3f> directions, std::vector<Eigen::Vector3f> normals, std::vector<Eigen::Vector3f> sampledColors, int rows, int cols);
    void importanceSampling(int rows, int cols, std::vector<Eigen::Vector3f>& directionVectors, std::vector<Eigen::Vector3f>& sampledColors, std::vector<Eigen::Vector3f> inpainting);
    std::vector<Eigen::Vector3f> replaceBrdf(std::vector<Eigen::Vector3f> inpainting, std::vector<Eigen::Vector3f> mask, std::vector<Eigen::Vector3f> normals,int rows, int cols);

    std::vector<Eigen::Vector3f> paintEnvMap(std::vector<Eigen::Vector3f> inpainting, std::vector<Eigen::Vector3f> mask, std::vector<Eigen::Vector3f> normals,int rows, int cols, std::vector<Eigen::Vector3f> desiredColors, Eigen::Vector2f highlight);
    void sampleSpecular(std::vector<Eigen::Vector3f> &image, std::vector<Eigen::Vector3f> mask, std::vector<Eigen::Vector3f> normals,int rows, int cols, std::vector<Eigen::Vector3f> highlights, std::vector<Eigen::Vector3f> lightcols, float specular);
    void addHighlightsToEnvmap(std::vector<Eigen::Vector3f> &image, std::vector<Eigen::Vector3f> mask, std::vector<Eigen::Vector3f> normals, int rows, int cols, std::vector<Eigen::Vector3f> desiredColors, Eigen::Vector2f highlight, float shiny);

    Eigen::Vector3f m_diffuse;
    Eigen::Vector3f m_specular;

    std::map<int, std::vector<int>> pixelToSampleIds;
    void writeCoefficientsToFile(std::string filename, Eigen::MatrixXf data, int num);
    void writeDesiredToFile(std::string filename, std::vector<float> data);
    void saveEnvmap(std::vector<Eigen::Vector3f> sampledColors);
    void getNewEnvmap(std::string filename, Eigen::VectorXf &envmapChannel);
    Vector3f brdf(Vector3f sampleDir, Vector3f V, Vector3f objectNormal, int type, Vector3f li);

    std::vector<Vector3f> basisSH(Vector3f normal, Vector3f lightColor);
    std::vector<std::vector<Vector3f>> reduceSHLights(std::vector<Vector3f> sampleDirs, std::vector<Vector3f> sampleColors);
    std::vector<int> us;
    std::vector<int> vs;
    Eigen::MatrixXf reds;
    Eigen::MatrixXf greens;
    Eigen::MatrixXf blues;

    Eigen::MatrixXf redSpecular;
    Eigen::MatrixXf greenSpecular;
    Eigen::MatrixXf blueSpecular;

    std::vector<Eigen::Vector3f> specularDirs;
    std::vector<std::vector<Vector3f>> lc;

    int m_solve = 0;
    int m_maskArea = 0;
    int sampleNum = 0;

    int brdf_type = 0;

    std::vector<SHSample> m_samples;
};

#endif // BRDFREPLACEMENT_H
