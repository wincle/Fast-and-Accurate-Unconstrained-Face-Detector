#ifndef _LEARNGAB_HPP
#define _LEARNGAB_HPP
#include "common.hpp"
#include "LearnDQT.hpp"
#include <opencv2/core/core.hpp>
#include "data.hpp"

/*
 * breif The Detector for face classification
 */
class GAB{
  public:
    /*
     * \breif Init Feature map
     * Not only generate feature map and also generate feature-coordinate map
     */
    GAB();
    /*
     * \breif Train Detector
     * Soft Cascade structure
     *
     * \param pos  positive dataset for training
     * \param neg  negative dataset for training
     */
    void LearnGAB(DataSet& pos, DataSet& neg);
    /*
     * \breif Store a stage
     *
     * \param feaId  vector saved features index trained in this stage
     * \param leftChild  vector saved tree structure trained in this stage
     * \param rightChild  vector saved tree structure trained in this stage
     * \param cutpoint  vector saved double thresholds of a feature
     * \param fit  vector saved leaves score trained in this stage
     * \param threshold  threshold for this stage
     */
    void SaveIter(vector<int> feaId, vector<int> leftChild, vector<int> rightChild, vector< vector<unsigned char> > cutpoint, vector<float> fit, float threshold);
    /*
     * \bref Save dector to file
     */
    void Save();
    /*
     * \breif Get the depth of the tree
     *
     * \param leftChild  vector saved tree structure trained in this stage
     * \param rightChild  vector saved tree structure trained in this stage
     * \param node  index of node in the tree
     */
    int CalcTreeDepth(vector<int> leftChild, vector<int> rightChild, int node = 0);
    /*
     * \breif Update scores
     * score will set to be 0 as the first time
     * score will be accumulated in every stages
     * it will call function TestSubTree going through to a leaf
     *
     * \param posFx[]  scores to be accumulated
     * \param fit  vector saved leaves score trained in this stage
     * \param cutpoint vector saved double thresholds of a feature
     * \param leftChild  vector saved tree structure trained in this stage
     * \param rightChild  vector saved tree structure trained in this stage
     * \param x  Feature Mat calculated by Pixel Mat
     */
    void TestDQT(float posFx[], vector<float> fit, vector< vector<unsigned char> > cutpoint, vector<int> leftChild, vector<int> rightChild, cv::Mat x);
    /*
     * \breif Go through the tree to get a leaf score
     * The function has a index indicate which img to go throw the tree,
     * be different with matlab code which going through the tree all imgs together.
     *
     * \param fit  ...
     * \param cutpoint  ...
     * \param x  Featue Mat
     * \param node  index of node in the tree
     * \param index  index of image
     * \param leftChild  ...
     * \param rightChild  ...
     */
    float TestSubTree(vector<float> fit,vector< vector<unsigned char> > cutpoint,cv::Mat x,int node,int index,vector<int> leftChild, vector<int> rightChild);
    /*
     * \breif Validate the region is a face or not
     * Go throw all the stages and accumulate the scores, 
     * only if the score passed all the threshold, judge it to be a face
     *
     * param test  the region to be test
     * param score  the score finally it got
     * param sIndex  the index of winSize
     */
    bool NPDClassify(cv::Mat test,float &score,int sIndex);
    /*
     * \breif Get the coordinates by feature id
     * the feature number is calculate by (objSize*objSize)*(objSize*objSize-1)/2
     * so if you have a feature id, use this function to get the coordinates
     * got the coordinates you can calculate the feature value in image.
     * here use two maps which store feature-coordinates
     *
     * \param feaid  Feature Id
     * \param x1  coordinate of point A.x
     * \param x2  coordinate of point A.y
     * \param y1  coordinate of point B.x
     * \param y2  coordinate of point B.y
     */
    void GetPoints(int feaid, int *x1, int *y1, int *x2, int *y2);
    /*
     * \breif Get the pixel index by feature id
     * same to be Previous, use this function to get two pixel index
     *
     * \param x  pixel index of point A
     * \param y  pixel index of point B
     */
    void GetPoints(int feaid, int *x, int *y);
    /*
     * \breif Mining Negative Samples
     * Use NextImage to get regions and than use NPDClassify to validate it's a face or not
     * Using a Mining rate to control travel speed
     * Using region_pool for openmp
     *
     * \param n  the negative size final condition
     * \param neg  negative dataset
     */
    void MiningNeg(const int n,DataSet& neg);
    /*
     * \breif Load a model to detector
     *
     * \param path  file path of model
     */
    void LoadModel(string path);
    /*
     * \breif Draw rect in a image
     *
     * \param img  the image need to be draw
     * \param rects  the box
     */
    cv::Mat Draw(cv::Mat& img, cv::Rect& rects);
  public:
    /* \breif indicate how many stages the dector have */
    int stages;
    /* \breif vectors contain the model */
    vector<int> treeIndex;
    vector<int> feaIds, leftChilds, rightChilds;
    vector<unsigned char> cutpoints;
    vector<float> fits;
    vector<float> thresholds;
    int numBranchNodes;
    /* \breif save the points of feature id */
    vector< vector<int> > points1x,points2x,points1y,points2y;
    /* \breif vector contain point-feature map */
    vector<int> lpoints;
    vector<int> rpoints;
    /* \breif A feature map used for speed up calculate feature */
    cv::Mat ppNpdTable;
    /* \breif model template size */
    int DetectSize;
};
#endif
