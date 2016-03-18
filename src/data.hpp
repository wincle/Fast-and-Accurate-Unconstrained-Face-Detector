#ifndef _DATA_HPP
#define _DATA_HPP
#include "common.hpp"
#include <vector>
#include <opencv2/core/core.hpp>

class DataSet {
  public:
    DataSet();
    static void LoadDataSet(DataSet& pos, DataSet& neg, int stages);
    void LoadPositiveDataSet(const std::string& positive, int stages);
    void LoadNegativeDataSet(const std::string& negative,const int pos_num,int stages);
    cv::Mat NextImage(int );
    void MoreNeg(int );
    void Remove(vector<int>);
    void ImgClear();
    void initWeights();
    cv::Mat ExtractPixel();
    void CalcWeight(int y, int maxWeight);
    void Clear();
  public:
    std::vector<cv::Mat> imgs;
    int size;
    int numPixels;
    int feaDims;

    float *W;
    float *Fx;
    
    //neg only
    std::vector<std::string> list;
    std::vector<cv::Mat> NegImgs;
    /* \breif array of current image to generate negative samples ,
     *      * set the size to be your cores num */
    int current_id[16];
    /* \breif array of location for travel negative images */
    int x[16];
    int y[16];
    /* \breif array of factors for resize negative images when traveling negative images */
    float factor[16];
    /* \breif array of step when traveling negative images */
    int step[16];
    /* \breif array of flip type when traveling negative images */
    int tranType[16];
    /* \breif array of window size  when traveling negative images */
    int win[16];


};
#endif
