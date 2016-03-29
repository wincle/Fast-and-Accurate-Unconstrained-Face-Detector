#include "common.hpp"
#include "data.hpp"
#include <opencv2/core/core.hpp>
/* \breif Wraper for call Detector */
class TrainDetector{
  public:
    /* 
     * \breif Training
     * Load the dataset first and load model if exit,
     * Training Detector next
     */
    void Train();
};
