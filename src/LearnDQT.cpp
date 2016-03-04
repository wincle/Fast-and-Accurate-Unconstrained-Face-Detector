#include "LearnDQT.hpp"

void DQT::LearnDQT(cv::Mat posX,cv::Mat negX, float posW[], float negW[], float posFx[], float negFx[], vector<int> posIndex,vector<int> negIndex, int minLeaf){
  const Options& opt = Options::GetInstance();
  int numThreads = omp_get_num_procs();
  int nTotalPos = posX.cols;
  int nTotalNeg = negX.cols;
  int feaDims = posX.rows;
  int nPos = posIndex.size();
  int nNeg = negIndex.size();

  vector<int> feaId, leftChild, rightChild;
  vector< vector<unsigned char> > cutpoint;
  vector<float> fit;


}
