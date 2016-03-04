#include "common.hpp"
#include <opencv2/core/core.hpp>
#include <omp.h>

class DQT{
  public:
    void LearnDQT(cv::Mat ,cv::Mat, float[], float[], float[], float[], vector<int> ,vector<int>, int);
};
