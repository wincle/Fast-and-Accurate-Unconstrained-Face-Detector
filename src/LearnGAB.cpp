#include "LearnGAB.hpp"
#include <math.h>
#include<sys/time.h>

#define max(a, b)  (((a) > (b)) ? (a) : (b))
#define min(a, b)  (((a) < (b)) ? (a) : (b))

using namespace cv;

GAB::GAB(){
  const Options& opt = Options::GetInstance();
  stages = 0;
  ppNpdTable = Mat(256,256,CV_8UC1);

  for(int i = 0; i < 256; i++)
  {
    for(int j = 0; j < 256; j++)
    {
      double fea = 0.5;
      if(i > 0 || j > 0) fea = double(i) / (double(i) + double(j));
      fea = floor(256 * fea);
      if(fea > 255) fea = 255;

      ppNpdTable.at<uchar>(i,j) = (unsigned char) fea;
    }
  }

  size_t numPixels = opt.objSize*opt.objSize;
  for(int i = 0; i < numPixels; i++)
  {
    for(int j = i+1; j < numPixels; j ++)
    {
      lpoints.push_back(i);
      rpoints.push_back(j);
    }
  } 

}

vector<int> GAB::LearnGAB(Mat& faceFea, Mat& nonfaceFea){
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
  vector<int> negPassIndex;
  negPassIndex.reserve(nNeg);
  for(int i=0; i<nPos; i++)
    negPassIndex.push_back(i);
  int nNegPass = nNeg;
//  printf("Start to train AdaBoost. nPos=%d, nNeg=%d\n\n", nPos, nNeg);

  float FAR=1.0;
  int nFea=0;
  float aveEval=0;

  for (int t = stages;t<opt.maxNumWeaks;t++){
    gettimeofday(&start,NULL);
    if (nNegPass < opt.minSamples){
      printf("\nNo enough negative samples. The AdaBoost learning terminates at iteration %d. nNegPass = %d.\n", t - 1, nNegPass);
      break;
    }
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
    negIndex.reserve(nNegPass);
    for(int i=0; i<nNegPass; i++)
      negIndex.push_back(i);
    for(int i=0; i<nNegPass; i++) {
      int j = rand()%(nNegPass-i)+i;
      swap(negIndex[j],negIndex[i]);
    }

    negIndex.erase(negIndex.begin()+nNegSam,negIndex.end());
    for(int i = 0;i < nNegSam;i++){
      negIndex[i] = negPassIndex[negIndex[i]];
    }
    
    //trim weight
    float *w = new float[nPos];
    memcpy(w,posW,nPos*sizeof(float));
    std::sort(&w[0],&w[nPos]);
    int k; 
    float wsum;
    for(int i =0;i<nPos;i++){
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

    wsum = 0;
    memcpy(w,negW,nNeg*sizeof(float));
    std::sort(&w[0],&w[nNeg]);
    for(int i =0;i<nNeg;i++){
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
    int minLeaf_t = max( round((nPosSam+nNegSam)*opt.minLeafFrac),opt.minLeaf);

    vector<int> feaId, leftChild, rightChild;
    vector< vector<unsigned char> > cutpoint;
    vector<float> fit;

    DQT dqt;
    float mincost = dqt.Learn(faceFea,nonfaceFea,posW,negW,posIndex,negIndex,minLeaf_t,feaId,leftChild,rightChild,cutpoint,fit);

    if (feaId.empty()){
      printf("\n\nNo available features to satisfy the split. The AdaBoost learning terminates.\n");
      break;
    }

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

    printf("Iter %d: nPos=%d, nNeg=%d, ", t, nPosSam, nNegSam);
    printf("FAR(t)=%.2f%%, FAR=%.2g, depth=%d, nFea(t)=%d, nFea=%d, cost=%.3f.\n",far*100.,FAR,depth,feaId.size(),nFea,mincost);
    printf("\t\tnNegPass=%d, aveEval=%.3f, time=%.0fs, meanT=%.3fs.\n", nNegPass, aveEval, time, time/(t-stages+1));

    
    if(FAR<=opt.maxFAR){
      printf("\n\nThe training is converged at iteration %d. FAR = %.2f%%\n", t, FAR * 100);
      break;
    }

    if (nNegPass < nNeg * opt.minNegRatio || nNegPass < opt.minSamples){
      printf("\n\nNo enough negative samples. The AdaBoost learning terminates at iteration %d. nNegPass = %d.\n", t, nNegPass);
      break;
    }

    SaveIter(feaId,leftChild,rightChild,cutpoint,fit,threshold,far,depth);
    free(w);

    CalcWeight(posW,posFx,1,opt.maxWeight,nPos);
    CalcWeight(negW,negFx,-1,opt.maxWeight,negPassIndex);

  }
  return negPassIndex;
}

void GAB::SaveIter(vector<int> feaId, vector<int> leftChild, vector<int> rightChild, vector< vector<unsigned char> > cutpoint, vector<float> fit, float threshold, float far, int depth){
  const Options& opt = Options::GetInstance();

  feaIds.push_back(feaId); 
  leftChilds.push_back(leftChild); 
  rightChilds.push_back(rightChild);
  cutpoints.push_back(cutpoint);
  fits.push_back(fit);
  depths.push_back(depth);
  thresholds.push_back(threshold);
  fars.push_back(far);
  stages++;

}
void GAB::Save(){
  const Options& opt = Options::GetInstance();
  FILE* file;
  file = fopen(opt.outFile.c_str(), "wb+");

  fwrite(&opt.objSize,sizeof(int),1,file);
  fwrite(&stages,sizeof(int),1,file);
  for(int i = 0;i<stages;i++){
    fwrite(&feaIds[i],sizeof(int),feaIds[i].size(),file);
    fwrite(&cutpoints[i],2*sizeof(unsigned char),cutpoints[i].size(),file);
    fwrite(&leftChilds[i],sizeof(int),leftChilds[i].size(),file);
    fwrite(&rightChilds[i],sizeof(int),rightChilds[i].size(),file);
    fwrite(&fits[i],sizeof(float),fits[i].size(),file);
    fwrite(&depths[i],sizeof(int),1,file);
    fwrite(&thresholds[i],sizeof(float),1,file);
  }
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

bool GAB::NPDClassify(Mat test){
  float Fx = 0;
  int x1,y1,x2,y2;
  for(int i = 0 ;i<stages;i++){
    int node = 0;
    while(node > -1){
      int feaid = feaIds[i][node];
      GetPoints(feaid,&x1,&y1,&x2,&y2);
      unsigned char p1 = test.at<uchar>(x1,y1);
      unsigned char p2 = test.at<uchar>(x2,y2);
      unsigned char fea = ppNpdTable.at<uchar>(p1,p2);

      if(fea < cutpoints[i][node][0] || fea > cutpoints[i][node][1])
        node = leftChilds[i][node];
      else
        node = rightChilds[i][node];
    }

    node = -node -1;
    Fx = Fx + fits[i][node];

    if(Fx < thresholds[i])
      return false;

  }
  return true;
}

void GAB::GetPoints(int feaid, int *x1, int *y1, int *x2, int *y2){
  const Options& opt = Options::GetInstance();
  int lpoint = lpoints[feaid];
  int rpoint = rpoints[feaid];
  *y1 = lpoint%opt.objSize;
  *x1 = lpoint/opt.objSize;
  *y2 = rpoint%opt.objSize;
  *x2 = rpoint/opt.objSize;
}
