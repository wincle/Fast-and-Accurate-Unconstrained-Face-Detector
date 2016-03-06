#include "LearnGAB.hpp"
#include <math.h>

#define max(a, b)  (((a) > (b)) ? (a) : (b))
#define min(a, b)  (((a) < (b)) ? (a) : (b))

using namespace cv;

GAB::GAB(){
}

void randperm(int n,vector<int> perm)
{
  int i, j, t;
  for(i=0; i<n; i++)
    perm.push_back(i);
  for(i=0; i<n; i++) {
    j = rand()%(n-i)+i;
    swap(perm[j],perm[i]);
  }
}

void GAB::LearnGAB(Mat faceFea, Mat nonfaceFea){
  const Options& opt = Options::GetInstance();
  int nPos = faceFea.cols;
  int nNeg = nonfaceFea.cols;
  float *posW = new float[nPos];
  for(int i = 0;i<nPos;i++)
    posW[i]=1./nPos;
  float *negW = new float[nNeg];
  for(int i = 0;i<nNeg;i++)
    negW[i]=1./nNeg;
  float *posFx = new float[nPos];
  float *negFx = new float[nNeg];
  for(int i = 0;i<nPos;i++)
        posFx[i]=0;
  for(int i = 0;i<nNeg;i++)
        negFx[i]=0;
  int startIter = 1;
  int *negPassIndex = new int[nNeg];
  for(int i = 0;i<nNeg;i++)
    negPassIndex[i]=i;
  int nNegPass = nNeg;
  printf("Start to train AdaBoost. nPos=%d, nNeg=%d\n\n", nPos, nNeg);
  for (int t = startIter;t<opt.maxNumWeaks;t++){
    if (nNegPass < opt.minSamples)
      printf("\nNo enough negative samples. The AdaBoost learning terminates at iteration %d. nNegPass = %d.\n", t - 1, nNegPass);

    int nPosSam = max(round(nPos * opt.samFrac),opt.minSamples);
    vector<int> posIndex;
    posIndex.reserve(nPos);
    for(int i=0; i<nPos; i++)
      posIndex.push_back(i);
    for(int i=0; i<nPos; i++) {
      int j = rand()%(nPos-i)+i;
      swap(posIndex[j],posIndex[i]);
    }

    int nNegSam = max(round(nNegPass * opt.samFrac), opt.minSamples);
    vector<int> negIndex;
    negIndex.reserve(nPos);
    for(int i=0; i<nPos; i++)
      negIndex.push_back(i);
    for(int i=0; i<nPos; i++) {
      int j = rand()%(nPos-i)+i;
      swap(negIndex[j],negIndex[i]);
    }
    
    //trim weight
    float *w = new float[nPos];
    memcpy(w,posW,nPos*sizeof(float));
    std::sort(&w[0],&w[nPos]);
    int k; 
    float wsum;
    for(int i =1;i<nPos;i++){
      wsum += w[i];
      if (wsum>=opt.trimFrac){
        k = i;
      }
    }
    k = min(k,nPosSam-opt.minSamples+1);
    vector< int >::iterator iter;
    for(iter = posIndex.begin();iter!=posIndex.end();){
      if(posW[*iter]<w[k])
        iter = posIndex.erase(iter);
      else
        ++iter;
    }

    memcpy(w,negW,nNeg*sizeof(float));
    std::sort(&w[0],&w[nNeg]);
    for(int i =1;i<nNeg;i++){
      wsum += w[i];
      if (wsum>=opt.trimFrac){
        k = i;
      }
    }
    k = min(k,nNegSam-opt.minSamples+1);
    for(iter = negIndex.begin();iter!=negIndex.end();){
      if(negW[*iter]<w[k])
        iter = negIndex.erase(iter);
      else
        ++iter;
    }



    nPosSam = posIndex.size();
    nNegSam = negIndex.size();
    int minLeaf_t = max( round((nPosSam+nNegSam+0.5)*opt.minLeafFrac),opt.minLeaf);
    printf("Iter %d: nPos=%d, nNeg=%d, ", t, nPosSam, nNegSam);

    vector<int> feaId, leftChild, rightChild;
    vector< vector<unsigned char> > cutpoint;
    vector<float> fit;

    DQT dqt;
    float mincost = dqt.Learn(faceFea,nonfaceFea,posW,negW,posFx,negFx,posIndex,negIndex,minLeaf_t,feaId,leftChild,rightChild,cutpoint,fit);




  }
}
