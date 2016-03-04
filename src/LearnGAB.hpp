#include "common.hpp"
#include "LearnDQT.hpp"
#include <opencv2/core/core.hpp>

class GAB{
  public:
    GAB();
    void LearnGAB(cv::Mat faceFea, cv::Mat nonfaceFea);
    vector<DQT> trees;
  private:
};
