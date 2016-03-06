#include "common.hpp"
#include <opencv2/core/core.hpp>
#include <omp.h>

class DQT{
  public:
    float Learn(cv::Mat ,cv::Mat, float[], float[], float[], float[], vector<int> ,vector<int>, int, vector<int>, vector<int>, vector<int>, vector< vector<unsigned char> >, vector<float>);
};
