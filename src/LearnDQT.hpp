#ifndef _LEARNDQT_HPP
#define _LEARNDQT_HPP
#include "common.hpp"
#include "data.hpp"
#include <opencv2/core/core.hpp>
#include <omp.h>
/*
 * \breif Stage Training Process
 */
class DQT{
  public:
    /* \breif feature-pixel map used for locate pixed location by featureId */
    vector<int> lpoints;
    vector<int> rpoints;
    /* \breif A feature map used for speed up calculate feature */
    cv::Mat ppNpdTable;

  public:
    /*
     * \breif Init maps
     */
    DQT();
    /*
     * \breif Main Preocess Wraper
     *
     * \param PosX  positive samples pixel information
     * \param NegX  negative samples pixel information
     * \param pPosW  positive samples weights
     * \param pNegW  negative samples weights
     * \param posIndex  positive samples index
     * \param negIndex  negative samples index
     * \param minLeaf  minimum leaf number
     * \param feaId  features learned by this stage
     * \param leftChild  tree structure learned by this stage
     * \param rightChild  tree structure learned by this stage
     * \param cutpoint  double thresholds learned by this stage
     * \param fit  score learned by this stage
     */
    float Learn(cv::Mat posX,cv::Mat negX, float pPosW[], float pNegW[], vector<int> posIndex,vector<int> negIndex, int minLeaf, vector<int> &feaId, vector<int> &leftChild, vector<int> &rightChild, vector< vector<unsigned char> > &cutpoint, vector<float> &fit);
    /*
     * \breif Learn one node in the tree
     * 
     * \param posX  ...
     * \param negX  ...
     * \param posW  ...
     * \param negW  ...
     * \param posIndex  ...
     * \param negIndex  ...
     * \param nPos  number of positive samples
     * \param nNeg  number of negative samples
     * \param minLeaf  ...
     * \param numThreads  openmp threads number
     * \param parentFit  parent score of the tree(default:0)
     * \param feaId  feature learned by this node
     * \param cutpoint  double thresholds learned by this node
     * \param fit  score learned by this node
     */
    float LearnQuadStump(vector<unsigned char *> &posX, vector<unsigned char *> &negX, float *posW, float *negW, int *posIndex, int *negIndex, int nPos, int nNeg, int minLeaf, int numThreads, float parentFit, int &feaId, unsigned char (&cutpoint)[2], float (&fit)[2]);
    /*
     * \breif Main Learning Process of a tree
     * \param posX  ...
     * \param negX  ...
     * \param posW  ...
     * \param negW  ...
     * \param posIndex  ...
     * \param negIndex  ...
     * \param nPos  ...
     * \param nNeg  ...
     * \param treeLevel  max tree depths
     * \param minLeaf  ...
     * \param numThreads  ...
     * \param parentFit  ...
     * \param feaId  ...
     * \param cutpoint  ...
     * \param fit  ...
     */
    float LearnDQT(vector<unsigned char *> &posX, vector<unsigned char *> &negX, float *posW, float *negW, int *posIndex, int *negIndex, int nPos, int nNeg, int treeLevel, int minLeaf, int numThreads, float parentFit, vector<int> &feaId, vector< vector<unsigned char> > &cutpoint, vector<int> &leftChild, vector<int> &rightChild, vector<float> &fit);
    /*
     * \breif Count Weights by features
     *
     * \param X  samples pixel at point A
     * \param Y  samples pixel at point B
     * \param index  index of samples
     * \param n  samples number
     * \param count  count pixel feature in this
     * \param wHist  count weights in this
     */
    void WeightHist(unsigned char *X, unsigned char *Y, float *W, int *index, int n, int count[256], float wHist[256]);
    /*
     * \breif Get pixel location by feature Id
     *
     * \param feaid  feature Id
     * \param x  point A location
     * \param y  point B location
     */
    void GetPoints(int feaid, int *x, int *y);
};
#endif
