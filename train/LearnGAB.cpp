#include "LearnGAB.hpp"
#include <math.h>
#include <sys/time.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <map>

#define max(a, b)  (((a) > (b)) ? (a) : (b))
#define min(a, b)  (((a) < (b)) ? (a) : (b))

using namespace cv;

int pWinSize[]={24,29,35,41,50,60,72,86,103,124,149,178,214,257,308,370,444,532,639,767,920,1104,1325,1590,1908,2290,2747,3297,3956};

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


  points1x.resize(29);
  points1y.resize(29);
  points2x.resize(29);
  points2y.resize(29);

  numBranchNodes = 0;
}

void GAB::LearnGAB(DataSet& pos, DataSet& neg){
  const Options& opt = Options::GetInstance();
  timeval start, end;
  timeval Tstart, Tend;
  float time = 0;
  int nPos = pos.size;
  int nNeg = neg.size;

  float _FAR=1.0;
  int nFea=0;
  float aveEval=0;

  float *wP = new float[nPos];
  float *wN = new float[nNeg];

  if(stages!=0){
    
    int fail = 0;
    #pragma omp parallel for
    for (int i = 0; i < nPos; i++) {
      float score = 0;
      if(NPDClassify(pos.imgs[i],score,0)){
          pos.Fx[i]=score;
      }
      else{
        fail ++;
      }
    }
    if(fail!=0){
      printf("you should't change pos data! %d \n",fail);
      return;
    }


    MiningNeg(nPos*opt.negRatio,neg);
    if(neg.imgs.size()<pos.imgs.size()*opt.negRatio){
      printf("neg not enough, change neg rate or add neg Imgs %d %d\n",pos.imgs.size(),neg.imgs.size());
      return;
    }

    pos.CalcWeight(1,opt.maxWeight);
    neg.CalcWeight(-1,opt.maxWeight);

  }

  Mat faceFea = pos.ExtractPixel();
  pos.ImgClear();
  printf("Extract pos feature finish\n");
  Mat nonfaceFea = neg.ExtractPixel();
  printf("Extract neg feature finish\n");

  for (int t = stages;t<opt.maxNumWeaks;t++){
    nNeg  = neg.size;
    printf("start training %d stages \n",t);
    gettimeofday(&start,NULL);

    vector<int> posIndex;
    vector<int> negIndex;
    for(int i=0; i<nPos; i++)
      posIndex.push_back(i);
    for(int i=0; i<nNeg; i++)
      negIndex.push_back(i);

    //trim weight
    memcpy(wP,pos.W,nPos*sizeof(float));
    std::sort(&wP[0],&wP[nPos]);
    int k; 
    float wsum;
    for(int i =0;i<nPos;i++){
      wsum += wP[i];
      if (wsum>=opt.trimFrac){
        k = i;
        break;
      }
    }
    vector< int >::iterator iter;
    for(iter = posIndex.begin();iter!=posIndex.end();){
      if(pos.W[*iter]<wP[k])
        iter = posIndex.erase(iter);
      else
        ++iter;
    }

    wsum = 0;
    memcpy(wN,neg.W,nNeg*sizeof(float));
    std::sort(&wN[0],&wN[nNeg]);
    for(int i =0;i<nNeg;i++){
      wsum += wN[i];
      if (wsum>=opt.trimFrac){
        k = i;
        break;
      }
    }
    for(iter = negIndex.begin();iter!=negIndex.end();){
      if(neg.W[*iter]<wN[k])
        iter = negIndex.erase(iter);
      else
        ++iter;
    }

    int nPosSam = posIndex.size();
    int nNegSam = negIndex.size();

    int minLeaf_t = max( round((nPosSam+nNegSam)*opt.minLeafFrac),opt.minLeaf);

    vector<int> feaId, leftChild, rightChild;
    vector< vector<unsigned char> > cutpoint;
    vector<float> fit;

    printf("Iter %d: nPos=%d, nNeg=%d, ", t, nPosSam, nNegSam);
    DQT dqt;
    gettimeofday(&Tstart,NULL);
    float mincost = dqt.Learn(faceFea,nonfaceFea,pos.W,neg.W,posIndex,negIndex,minLeaf_t,feaId,leftChild,rightChild,cutpoint,fit);
    gettimeofday(&Tend,NULL);
    float DQTtime = (Tend.tv_sec - Tstart.tv_sec);
    printf("DQT time:%.3fs\n",DQTtime);

    if (feaId.empty()){
      printf("\n\nNo available features to satisfy the split. The AdaBoost learning terminates.\n");
      break;
    }

    Mat posX(feaId.size(),faceFea.cols,CV_8UC1);
    for(int i = 0;i<feaId.size();i++)
      for(int j = 0;j<faceFea.cols;j++){
        int x,y;
        GetPoints(feaId[i],&x,&y);
        unsigned char Fea = ppNpdTable.at<uchar>(faceFea.at<uchar>(x,j),faceFea.at<uchar>(y,j));
        posX.at<uchar>(i,j) = Fea;
      }
    Mat negX(feaId.size(),nonfaceFea.cols,CV_8UC1);
    for(int i = 0;i<feaId.size();i++)
      for(int j = 0;j<nonfaceFea.cols;j++){
        int x,y;
        GetPoints(feaId[i],&x,&y);
        unsigned char Fea = ppNpdTable.at<uchar>(nonfaceFea.at<uchar>(x,j),nonfaceFea.at<uchar>(y,j));
        negX.at<uchar>(i,j) = Fea;
      }

    TestDQT(pos.Fx,fit,cutpoint,leftChild,rightChild,posX);
    TestDQT(neg.Fx,fit,cutpoint,leftChild,rightChild,negX);
    

    vector<int> negPassIndex;
    for(int i=0; i<nNegSam; i++)
      negPassIndex.push_back(i);

    memcpy(wP,pos.Fx,nPos*sizeof(float));
    sort(wP,wP+nPos);
    int index = max(floor(nPos*(1-opt.minDR)),0);
    float threshold = wP[index];

    for(iter = negPassIndex.begin(); iter != negPassIndex.end();){
      if(neg.Fx[*iter] < threshold)
        iter = negPassIndex.erase(iter);
      else
        iter++;
    }
    float far = float(negPassIndex.size())/float(nNeg);

  
    int depth = CalcTreeDepth(leftChild,rightChild);

    if(t==1)
      aveEval+=depth;
    else
      aveEval+=depth*_FAR;
    _FAR *=far;
    nFea = nFea + feaId.size();


    gettimeofday(&end,NULL);
    time += (end.tv_sec - start.tv_sec);

    int nNegPass = negPassIndex.size();
    printf("FAR(t)=%.2f%%, FAR=%.2g, depth=%d, nFea(t)=%d, nFea=%d, cost=%.3f.\n",far*100.,_FAR,depth,feaId.size(),nFea,mincost);
    printf("\t\tnNegPass=%d, aveEval=%.3f, time=%.3fs, meanT=%.3fs.\n", nNegPass, aveEval, time, time/(stages+1));

    
    if(_FAR<=opt.maxFAR){
      printf("\n\nThe training is converged at iteration %d. FAR = %.2f%%\n", t, _FAR * 100);
      break;
    }


    SaveIter(feaId,leftChild,rightChild,cutpoint,fit,threshold);

    gettimeofday(&Tstart,NULL); 

    neg.Remove(negPassIndex);
    if(neg.size<opt.minSamples)
      MiningNeg(nPos*opt.negRatio,neg);
   
    nonfaceFea = neg.ExtractPixel();
    pos.CalcWeight(1,opt.maxWeight);
    neg.CalcWeight(-1,opt.maxWeight);
    
    gettimeofday(&Tend,NULL);
    float Ttime = (Tend.tv_sec - Tstart.tv_sec);
    printf("neg mining time:%.3fs\n",Ttime);

    if(!(stages%opt.saveStep)){
      Save();
      printf("save the model\n");
    }

  }
  delete []wP;
  delete []wN;

}

void GAB::SaveIter(vector<int> feaId, vector<int> leftChild, vector<int> rightChild, vector< vector<unsigned char> > cutpoint, vector<float> fit, float threshold){
  const Options& opt = Options::GetInstance();

  int root = numBranchNodes;
  treeIndex.push_back(root);
  numBranchNodes += feaId.size();

  for(int i = 0;i<feaId.size();i++){
    feaIds.push_back(feaId[i]);
    for(int j = 0;j<29;j++){
      int x1,y1,x2,y2;
      GetPoints(feaId[i],&x1,&y1,&x2,&y2);
      float factor = (float)pWinSize[j]/(float)opt.objSize;
      points1x[j].push_back(x1*factor);
      points1y[j].push_back(y1*factor);
      points2x[j].push_back(x2*factor);
      points2y[j].push_back(y2*factor);
    }
    if(leftChild[i]<0)
      leftChild[i] -= (treeIndex[stages]+stages);
    else
      leftChild[i] += treeIndex[stages];
    leftChilds.push_back(leftChild[i]); 
    if(rightChild[i]<0)
      rightChild[i] -= (treeIndex[stages]+stages);
    else
      rightChild[i] += treeIndex[stages];
    rightChilds.push_back(rightChild[i]);
    cutpoints.push_back(cutpoint[i][0]);
    cutpoints.push_back(cutpoint[i][1]);
  }
  for(int i = 0;i<fit.size();i++)
    fits.push_back(fit[i]);
  thresholds.push_back(threshold);
  stages++;
  
}
void GAB::Save(){
  const Options& opt = Options::GetInstance();
  FILE* file;
  file = fopen(opt.outFile.c_str(), "wb");

  fwrite(&opt.objSize,sizeof(int),1,file);
  fwrite(&stages,sizeof(int),1,file);
  fwrite(&numBranchNodes,sizeof(int),1,file);
  
  int *tree = new int[stages];
  float *threshold = new float[stages];
  for(int i = 0;i<stages;i++){
    tree[i] = treeIndex[i];
    threshold[i] = thresholds[i];
  }
  fwrite(tree,sizeof(int),stages,file);
  fwrite(threshold,sizeof(float),stages,file);
  delete []tree;
  delete []threshold;

  int *feaId = new int[numBranchNodes];
  int *leftChild = new int[numBranchNodes];
  int *rightChild = new int[numBranchNodes];
  unsigned char* cutpoint = new unsigned char[2*numBranchNodes];
  for(int i = 0;i<numBranchNodes;i++){
    feaId[i] = feaIds[i];
    leftChild[i] = leftChilds[i];
    rightChild[i] = rightChilds[i];
    cutpoint[2*i] = cutpoints[i*2];
    cutpoint[2*i+1] = cutpoints[i*2+1];
  }
  fwrite(feaId,sizeof(int),numBranchNodes,file);
  fwrite(leftChild,sizeof(int),numBranchNodes,file);
  fwrite(rightChild,sizeof(int),numBranchNodes,file);
  fwrite(cutpoint,sizeof(unsigned char),2*numBranchNodes,file);
  delete []feaId;
  delete []leftChild;
  delete []rightChild;
  delete []cutpoint;

  int numLeafNodes = numBranchNodes+stages;
  float *fit = new float[numLeafNodes];
  for(int i = 0;i<numLeafNodes;i++)
    fit[i] = fits[i];
  fwrite(fit,sizeof(float),numLeafNodes,file);
  delete []fit;

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

  #pragma omp parallel for
  for( int i = 0; i<n;i++)
    score[i] = TestSubTree(fit,cutpoint,x,0,i,leftChild,rightChild);

  for(int i =0;i<n;i++)
    posFx[i]+=score[i];

  delete []score;
}

float GAB::TestSubTree(vector<float> fit, vector< vector<unsigned char> > cutpoint, cv::Mat x, int node, int index, vector<int> leftChild, vector<int> rightChild){
  int n = x.cols;
  float score = 0;

  if (node<0){
    score=fit[-node-1];
  }

  else{
    bool isLeft;
    if(x.at<uchar>(node,index)<cutpoint[node][0] || x.at<uchar>(node,index)>cutpoint[node][1])
      isLeft = 1;
    else
      isLeft = 0;

    if(isLeft)
      score = TestSubTree(fit,cutpoint,x,leftChild[node],index,leftChild,rightChild);
    else
      score = TestSubTree(fit,cutpoint,x,rightChild[node],index,leftChild,rightChild);
  }
  return score;
}

bool GAB::NPDClassify(Mat test,float &score, int sIndex){
  const Options& opt = Options::GetInstance();
  float Fx = 0;

  for(int i = 0 ;i<stages;i++){
    int node = treeIndex[i];
    while(node > -1){
      unsigned char p1 = test.at<uchar>(points1x[sIndex][node],points1y[sIndex][node]);
      unsigned char p2 = test.at<uchar>(points2x[sIndex][node],points2y[sIndex][node]);
      unsigned char fea = ppNpdTable.at<uchar>(p1,p2);

      if(fea < cutpoints[node*2] || fea > cutpoints[node*2+1])
        node = leftChilds[node];
      else
        node = rightChilds[node];
    }

    node = -node -1;
    Fx = Fx + fits[node];

    if(Fx < thresholds[i]){
      return 0;
    }
  }
  score = Fx;
  return 1;
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

void GAB::GetPoints(int feaid, int *x, int *y){
  const Options& opt = Options::GetInstance();
  int lpoint = lpoints[feaid];
  int rpoint = rpoints[feaid];
  *x = lpoint;
  *y = rpoint;
}

void GAB::MiningNeg(int n,DataSet& neg){
  const Options& opt = Options::GetInstance();
  int pool_size = opt.numThreads;
  vector<Mat> region_pool(pool_size);
  int st = neg.imgs.size();
  int all = 0;
  int need = n - st;
  double rate;

  while(st<n){
    #pragma omp parallel for
    for(int i = 0;i<pool_size;i++){
      region_pool[i] = neg.NextImage(i);
    }

    #pragma omp parallel for
    for (int i = 0; i < pool_size; i++) {
      float score = 0;
      if(NPDClassify(region_pool[i],score,0)){
        #pragma omp critical 
        {
          if(st%(n/10)==0)
            printf("%d get\n",st);
          neg.imgs.push_back(region_pool[i].clone());
          neg.Fx[st]=score;
          if(opt.generate_hd){
            char di[256];
            sprintf(di,"../data/hd/%d.jpg",st);
            imwrite(di,region_pool[i].clone());
          }
          st++;
        }
      }
      all++;
    }
  }
  neg.size = n;
  rate = ((double)(need))/(double)all;
  printf("mining success rate %lf\n",rate);
}

void GAB::LoadModel(string path){
  FILE* file;
  if((file = fopen(path.c_str(), "rb"))==NULL)
    return;

  fread(&DetectSize,sizeof(int),1,file);
  fread(&stages,sizeof(int),1,file);
  fread(&numBranchNodes,sizeof(int),1,file);
  printf("stages num :%d\n",stages);

  int *_tree = new int[stages];
  float *_threshold = new float[stages];
  fread(_tree,sizeof(int),stages,file);
  fread(_threshold,sizeof(float),stages,file);
  for(int i = 0;i<stages;i++){
    treeIndex.push_back(_tree[i]);
    thresholds.push_back(_threshold[i]);
  }
  delete []_tree;
  delete []_threshold;

  int *_feaId = new int[numBranchNodes];
  int *_leftChild = new int[numBranchNodes];
  int *_rightChild = new int[numBranchNodes];
  unsigned char* _cutpoint = new unsigned char[2*numBranchNodes];
  fread(_feaId,sizeof(int),numBranchNodes,file);
  fread(_leftChild,sizeof(int),numBranchNodes,file);
  fread(_rightChild,sizeof(int),numBranchNodes,file);
  fread(_cutpoint,sizeof(unsigned char),2*numBranchNodes,file);
  for(int i = 0;i<numBranchNodes;i++){
    feaIds.push_back(_feaId[i]);
    leftChilds.push_back(_leftChild[i]);
    rightChilds.push_back(_rightChild[i]);
    cutpoints.push_back(_cutpoint[2*i]);
    cutpoints.push_back(_cutpoint[2*i+1]);
    for(int j = 0;j<29;j++){
      int x1,y1,x2,y2;
      GetPoints(_feaId[i],&x1,&y1,&x2,&y2);
      float factor = (float)pWinSize[j]/(float)DetectSize;
      points1x[j].push_back(x1*factor);
      points1y[j].push_back(y1*factor);
      points2x[j].push_back(x2*factor);
      points2y[j].push_back(y2*factor);
    }
  }
  delete []_feaId;
  delete []_leftChild;
  delete []_rightChild;
  delete []_cutpoint;

  int numLeafNodes = numBranchNodes+stages;
  float *_fit = new float[numLeafNodes];
  fread(_fit,sizeof(float),numLeafNodes,file);
  for(int i = 0;i<numLeafNodes;i++){
    fits.push_back(_fit[i]);
  }
  delete []_fit;

  fclose(file);
}
