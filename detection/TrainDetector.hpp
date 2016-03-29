#include "common.hpp"
#include <opencv2/core/core.hpp>
/* \breif Wraper for call Detector */
class TrainDetector{
  public:
    /*
     * \breif Detect For FDDB
     */
    void FddbDetect();
    /*
     * \breif Detect face from camera
     */
    void Live();
};
