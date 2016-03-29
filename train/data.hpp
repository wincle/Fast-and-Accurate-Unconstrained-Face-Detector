#ifndef _DATA_HPP
#define _DATA_HPP
#include "common.hpp"
#include <vector>
#include <opencv2/core/core.hpp>

/*!
 * \breif DataSet Wrapper
 */
class DataSet {
  public:
    DataSet();
    /*
     * \breif Load Wrapper for `LoadPositiveDataSet` and `LoadNegative DataSet`
     * Since positive dataset and negative dataset may share some information between
     *  each other, we need to load them all together
     *
     * \param stages  indicate if it is resume.
     */
    static void LoadDataSet(DataSet& pos, DataSet& neg, int stages);
    /*
     * \breif Load Postive DataSet
     * All positive samples are listed in this text file with each line represents a sample.
     * dataset could be augment in this stage, it include minor, resize and shift.
     * finally it will be Produce ten times DataSet than origin.
     *
     * \param positive  a text file path
     * \param stages  indicate if it is resume
     */ 
    void LoadPositiveDataSet(const std::string& positive, int stages);
    /*
     * \breif Load Negative DataSet
     * We generate negative samples like positive samples before the program runs. Each line
     * of the text file hold another text file which holds the real negative sample path in
     * the filesystem, all the image's path will be saved and imgs will be generated in
     * these images.
     *
     * \param negative  a text file path
     * \param pos_num  the size of negative imgs should be
     * \param stages  indicate if it is resume
     */
    void LoadNegativeDataSet(const std::string& negative,const int pos_num,int stages);
    /*
     * \breif Generate a img for negative samples
     * Will scrach a negative samples from image which lived in list
     * the x,y and width,height of sample is random, it will be resized
     * finally will flip or minor the sample by random
     *
     * \param i  the index of image in list and seed for random
     */
    cv::Mat NextImage(int seed);
    /*
     * \breif Generate images for negative samples
     * Generate a pool to generate negative sample, in order to call NextImage in parallel
     *
     * param n  the number of samples needed
     */
    void MoreNeg(int n);
    /*
     * \breif Remove Images from negative samples
     * It delete images in the vector(imgs) and clear scores(Fx)
     * finally update the size
     *
     * param PassIndex  the index of images which should be keeped
     */
    void Remove(vector<int> PassIndex);
    /*
     * \breif Release the memory of W and Fx which are created by new
     */
    void ImgClear();
    /*
     * \breif Init weights and scores
     */
    void initWeights();
    /*
     * \breif Extract Imgs Information
     * Extract images from vector to a Mat
     * the rows should be the Pixel amount, the cols should be the images amount
     */
    cv::Mat ExtractPixel();
    /*
     * \breif Update Weight
     * Init Weight to 1/size
     * Calculate by exp(-y*Fx)
     *
     * param y  flag of dataset, pos is 1, neg is -1
     * param maxWeight  the upper limit of weights
     */
    void CalcWeight(int y, int maxWeight);
    /*
     * \breif Release the memory of vector imgs
     */
    void Clear();
  public:
    /* \breif samples for training */
    std::vector<cv::Mat> imgs;
    /* \breif number of samples */
    int size;
    /* \breif number of pixel in a sample */
    int numPixels;
    /* \breif number of features contrain in a sample */
    int feaDims;
    /* \breif Weights of samples */
    float *W;
    /* \breif Scores of samples */
    float *Fx;
  public:
    /* \breif stored paths of negative images */
    std::vector<std::string> list;
    /* \breif image pool for generate negative samples */
    std::vector<cv::Mat> NegImgs;
    /* \breif array of current image to generate negative samples ,
     * set the size to be your cores num */
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
