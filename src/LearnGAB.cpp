#include "LearnGAB.hpp"
#include <math.h>
#include<sys/time.h>

#define max(a, b)  (((a) > (b)) ? (a) : (b))
#define min(a, b)  (((a) < (b)) ? (a) : (b))

using namespace cv;

void GAB::LearnGAB(Mat faceFea, Mat nonfaceFea){
  const Options& opt = Options::GetInstance();
  timeval start, end;
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
  vector<int> negPassIndex;
  negPassIndex.reserve(nNeg);
  for(int i=0; i<nPos; i++)
    negPassIndex.push_back(i);
  int nNegPass = nNeg;
  printf("Start to train AdaBoost. nPos=%d, nNeg=%d\n\n", nPos, nNeg);

  FILE* file = fopen(opt.outFile.c_str(), "wb");
  fclose(file);

  float FAR=1.0;
  int nFea=0;
  float aveEval=0;

  for (int t = startIter;t<opt.maxNumWeaks;t++){
    gettimeofday(&start,NULL);
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
    posIndex.erase(posIndex.begin()+nPosSam,posIndex.end());

    int nNegSam = max(round(nNegPass * opt.samFrac), opt.minSamples);
    vector<int> negIndex;
    negIndex.reserve(nPos);
    for(int i=0; i<nPos; i++)
      negIndex.push_back(i);
    for(int i=0; i<nPos; i++) {
      int j = rand()%(nPos-i)+i;
      swap(negIndex[j],negIndex[i]);
    }

    negIndex.erase(negIndex.begin()+nNegSam,negIndex.end());
    for(int i = 0;i < nPosSam;i++){
      negIndex[i] = negPassIndex[negIndex[i]];
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
        break;
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
        break;
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

    Mat posX(feaId.size(),faceFea.cols,CV_8UC1);
    for(int i = 0;i<feaId.size();i++)
      for(int j = 0;j<faceFea.cols;j++)
        posX.at<uchar>(i,j) = faceFea.at<uchar>(feaId[i],j);
    Mat negX(feaId.size(),nonfaceFea.cols,CV_8UC1);
    for(int i = 0;i<feaId.size();i++)
      for(int j = 0;j<nonfaceFea.cols;j++)
        negX.at<uchar>(i,j) = nonfaceFea.at<uchar>(feaId[i],j);

    TestDQT(posFx,fit,cutpoint,leftChild,rightChild,posX);
    TestDQT(negFx,fit,cutpoint,leftChild,rightChild,negX,negPassIndex);
    
    memcpy(w,posFx,nPos*sizeof(float));
    sort(w,w+nPos);
    int index = max(floor(nPos*(1-opt.minDR)),1);
    float threshold = w[index];

    for(iter = negPassIndex.begin(); iter != negPassIndex.end();){
      if(negFx[*iter] < threshold)
        iter = negPassIndex.erase(iter);
      else
        iter++;
    }
    float far = float(negPassIndex.size())/float(nNegPass);
    nNegPass = negPassIndex.size();
  
    int depth = CalcTreeDepth(leftChild,rightChild);

    if(t==1)
      aveEval+=depth;
    else
      aveEval+=depth*FAR;
    FAR *=far;
    nFea = nFea + feaId.size();


    gettimeofday(&end,NULL);
    int time = (end.tv_sec - start.tv_sec);

    printf("FAR(t)=%.2f%%, FAR=%.2g, depth=%d, nFea(t)=%d, nFea=%d, cost=%.3f.\n",far*100.,FAR,depth,feaId.size(),nFea,mincost);
    printf("\t\tnNegPass=%d, aveEval=%.3f, time=%.0fs, meanT=%.3fs.\n", nNegPass, aveEval, time, time/(t-startIter+1));

    
    if(FAR<=opt.maxFAR){
      printf("\n\nThe training is converged at iteration %d. FAR = %.2f%%\n", t, FAR * 100);
      break;
    }

    if (nNegPass < nNeg * opt.minNegRatio || nNegPass < opt.minSamples){
      printf("\n\nNo enough negative samples. The AdaBoost learning terminates at iteration %d. nNegPass = %d.\n", t, nNegPass);
      break;
    }

    save_cascade_to_file(feaId,leftChild,rightChild,cutpoint,fit,threshold,far,depth);
    free(w);

    CalcWeight(posW,posFx,1,opt.maxWeight,nPos);
    CalcWeight(negW,negFx,-1,opt.maxWeight,negPassIndex);

  }
}

void GAB::save_cascade_to_file(vector<int> feaId, vector<int> leftChild, vector<int> rightChild, vector< vector<unsigned char> > cutpoint, vector<float> fit, float threshold, float far, int depth){
  const Options& opt = Options::GetInstance();

  FILE* file;
  file = fopen(opt.outFile.c_str(), "wb+");

  fwrite(&feaId,sizeof(int),feaId.size(),file);
  fwrite(&cutpoint,2*sizeof(unsigned char),cutpoint.size(),file);
  fwrite(&leftChild,sizeof(int),leftChild.size(),file);
  fwrite(&rightChild,sizeof(int),rightChild.size(),file);
  fwrite(&fit,sizeof(float),fit.size(),file);
  fwrite(&depth,sizeof(int),1,file);
  fwrite(&threshold,sizeof(float),1,file);
  fwrite(&far,sizeof(float),1,file);

  fclose(file);

}

int GAB::CalcTreeDepth(vector<int> leftChild, vector<int> rightChild, int node){
  int depth = 0;
  int ld,rd;
  if ((node+1)>leftChild.size())
    return depth;
  if (leftChild[node]<0)
    ld = 0;
  else
    ld = CalcTreeDepth(leftChild,rightChild,leftChild[node]);

  if (rightChild[node] < 0)
    rd = 0;
  else
    rd = CalcTreeDepth(leftChild, rightChild, rightChild[node]);

  depth = max(ld,rd) + 1;
  return depth;
}

void GAB::TestDQT(float posFx[], vector<float> fit, vector< vector<unsigned char> > cutpoint, vector<int> leftChild, vector<int> rightChild, cv::Mat x){
  int n = x.cols;
  float *score = new float[n];
  for (int i = 0;i<n;i++)
    score[i]=0;

  for( int i = 0; i<n;i++)
    score[i] = TestSubTree(fit,cutpoint,x,-1,i,leftChild,rightChild,0);

  for(int i =0;i<n;i++)
    posFx[i]+=score[i];
}

void GAB::TestDQT(float posFx[], vector<float> fit, vector< vector<unsigned char> > cutpoint, vector<int> leftChild, vector<int> rightChild, cv::Mat x, vector<int> negPassIndex){
  int n = negPassIndex.size();
  float *score = new float[n];
  for (int i = 0;i<n;i++)
    score[negPassIndex[i]]=0;

  for( int i = 0; i<n;i++){
    score[negPassIndex[i]] = TestSubTree(fit,cutpoint,x,-1,negPassIndex[i],leftChild,rightChild,0);
  }

  for(int i =0;i<n;i++)
    posFx[negPassIndex[i]]+=score[negPassIndex[i]];
}

float GAB::TestSubTree(vector<float> fit, vector< vector<unsigned char> > cutpoint, cv::Mat x, int node, int index, vector<int> leftChild, vector<int> rightChild,bool init){
  int n = x.cols;
  float score = 0;

  if (init && node<0){
    score=fit[-node-1];
  }

  else{
    node++;
    bool isLeft;
    if(x.at<uchar>(node,index)<cutpoint[node][0] || x.at<uchar>(node,index)>cutpoint[node][1])
      isLeft = 1;
    else
      isLeft = 0;

    if(isLeft)
      score = TestSubTree(fit,cutpoint,x,leftChild[node],index,leftChild,rightChild,1);
    else
      score = TestSubTree(fit,cutpoint,x,rightChild[node],index,leftChild,rightChild,1);
  }
  return score;
}

void GAB::CalcWeight(float F[], float Fx[], int y, int maxWeight, int nPos){
  float s = 0;
  for(int i = 0;i<nPos;i++){
    F[i]=min(exp(-y*Fx[i]),maxWeight);
    s += F[i];
  }
  if (s == 0)
    for(int i = 0;i<nPos;i++)
      F[i]=1/nPos;
  else
    for(int i = 0;i<nPos;i++)
      F[i]/=s;
}

void GAB::CalcWeight(float F[], float Fx[], int y, int maxWeight, vector<int> negPassIndex){
  float s = 0;
  for(int i = 0;i<negPassIndex.size();i++){
    F[negPassIndex[i]]=min(exp(-y*Fx[negPassIndex[i]]),maxWeight);
    s += F[negPassIndex[i]];
  }
  if (s == 0)
    for(int i = 0;i<negPassIndex.size();i++)
      F[negPassIndex[i]]=1/negPassIndex.size();
  else
    for(int i = 0;i<negPassIndex.size();i++)
      F[negPassIndex[i]]/=s;
}
