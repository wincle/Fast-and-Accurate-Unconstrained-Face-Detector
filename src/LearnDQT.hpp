#ifndef _LEARNDQT_HPP
#define _LEARNDQT_HPP
#include "common.hpp"
#include <opencv2/core/core.hpp>
#include <omp.h>

class DQT{
  public:
    float Learn(cv::Mat posX,cv::Mat negX, float pPosW[], float pNegW[], vector<int> posIndex,vector<int> negIndex, int minLeaf, vector<int> &feaId, vector<int> &leftChild, vector<int> &rightChild, vector< vector<unsigned char> > &cutpoint, vector<float> &fit);
    float LearnQuadStump(vector<unsigned char *> &posX, vector<unsigned char *> &negX, float *posW, float *negW, int *posIndex, int *negIndex, int nPos, int nNeg, int minLeaf, int numThreads, float parentFit, int &feaId, unsigned char (&cutpoint)[2], float (&fit)[2]);
    float LearnDQT(vector<unsigned char *> &posX, vector<unsigned char *> &negX, float *posW, float *negW, int *posIndex, int *negIndex, int nPos, int nNeg, int treeLevel, int minLeaf, int numThreads, float parentFit, vector<int> &feaId, vector< vector<unsigned char> > &cutpoint, vector<int> &leftChild, vector<int> &rightChild, vector<float> &fit);
    void WeightHist(unsigned char *X, float *W, int *index, int n, int count[256], float wHist[256]);
};
#endif
