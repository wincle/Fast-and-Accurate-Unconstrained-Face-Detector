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
    cv::Mat NextImage(int);
  public:
    std::vector<cv::Mat> imgs;
    bool is_pos;
    int size;
  private:
    //for neg
    std::vector<std::string> list;
    int current_idx;
    cv::Mat img;
};
