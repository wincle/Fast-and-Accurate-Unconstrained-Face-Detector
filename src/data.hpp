#include "common.hpp"
#include <vector>
#include <opencv2/core/core.hpp>

class DataSet {
  public:
    DataSet();
    static void LoadDataSet(DataSet& pos, DataSet& neg);
    void LoadPositiveDataSet(const std::string& positive);
    void LoadNegativeDataSet(const std::string& negative,const int pos_num);
    void MoreNeg(int );
    cv::Mat NextImage();
    void NextState();
  public:
    std::vector<cv::Mat> imgs;
    bool is_pos;
    int size;
  private:
    //for neg
    std::vector<std::string> list;
    typedef enum {
      ORIGIN = 0,
      ORIGIN_R,
      ORIGIN_RR,
      ORIGIN_RRR,
      ORIGIN_FLIP,
      ORIGIN_FLIP_R,
      ORIGIN_FLIP_RR,
      ORIGIN_FLIP_RRR,
    } TransformType;
    TransformType transform_type;
    int x,y;
    int current_idx;

};
