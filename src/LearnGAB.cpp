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

  float *w = new float[nPos];

  if(stages!=0){
    
    int fail = 0;
    #pragma omp parallel for
    for (int i = 0; i < nPos; i++) {
      float score = 0;
      if(NPDClassify(pos.imgs[i].clone(),score,0)){
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


    MiningNeg(nPos,neg);
    if(neg.imgs.size()<pos.imgs.size()){
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
    memcpy(w,pos.W,nPos*sizeof(float));
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
    vector< int >::iterator iter;
    for(iter = posIndex.begin();iter!=posIndex.end();){
      if(pos.W[*iter]<w[k])
        iter = posIndex.erase(iter);
      else
        ++iter;
    }

    wsum = 0;
    memcpy(w,neg.W,nNeg*sizeof(float));
    std::sort(&w[0],&w[nNeg]);
    for(int i =0;i<nNeg;i++){
      wsum += w[i];
      if (wsum>=opt.trimFrac){
        k = i;
        break;
      }
    }
    for(iter = negIndex.begin();iter!=negIndex.end();){
      if(neg.W[*iter]<w[k])
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

    memcpy(w,pos.Fx,nPos*sizeof(float));
    sort(w,w+nPos);
    int index = max(floor(nPos*(1-opt.minDR)),0);
    float threshold = w[index];

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
      MiningNeg(nPos,neg);
   
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
  delete []w;

}

void GAB::SaveIter(vector<int> feaId, vector<int> leftChild, vector<int> rightChild, vector< vector<unsigned char> > cutpoint, vector<float> fit, float threshold){
  const Options& opt = Options::GetInstance();

  int root = 0;
  if(stages == 0)
    treeIndex.push_back(0);
  else
    root = treeIndex[stages];
  root += feaId.size();
  treeIndex.push_back(root);

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
  numBranchNodes = treeIndex[stages];
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

  //  #pragma omp parallel for
    for (int i = 0; i < pool_size; i++) {
      float score = 0;
      if(NPDClassify(region_pool[i].clone(),score,0)){
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

/*
void GAB::LoadModel(string path){
  FILE* file;
  if((file = fopen(path.c_str(), "rb"))==NULL)
    return;
  int size;

  points1x.resize(29);
  points1y.resize(29);
  points2x.resize(29);
  points2y.resize(29);

  int pWinSize[29]={24,29,35,41,50,60,72,86,103,124,149,178,214,257,308,370,444,532,639,767,920,1104,1325,1590,1908,2290,2747,3297,3956};

  fread(&DetectSize,sizeof(int),1,file);
  fread(&stages,sizeof(int),1,file);
  printf("stages num :%d\n",stages);
  treeIndex.push_back(0);
  for(int j = 0;j<stages;j++){
    printf("stage: %d\n",j);

    fread(&size,sizeof(int),1,file);
    int root = treeIndex[j];
    root += size;
    treeIndex.push_back(root);

    int *_feaId = new int[size];
    fread(_feaId,sizeof(int),size,file);
    for(int i = 0;i<size;i++){
      feaIds.push_back(_feaId[i]);
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
    unsigned char *_cutpoint = new unsigned char[size*2];
    fread(_cutpoint,sizeof(unsigned char),2*size,file);
    for(int i =0;i<size;i++){
      cutpoints.push_back(_cutpoint[2*i]);
      cutpoints.push_back(_cutpoint[2*i+1]);
    }
    int *_leftChild = new int[size];
    fread(_leftChild,sizeof(int),size,file);
    for(int i =0;i<size;i++){
      if(_leftChild[i]<0)
        _leftChild[i] -= (treeIndex[j]+j);
      else
        _leftChild[i] += treeIndex[j];
      leftChilds.push_back(_leftChild[i]);
    }
    int *_rightChild = new int[size];
    fread(_rightChild,sizeof(int),size,file);
    for(int i =0;i<size;i++){
      if(_rightChild[i]<0)
        _rightChild[i] -= (treeIndex[j]+j);
      else
        _rightChild[i] += treeIndex[j];
      rightChilds.push_back(_rightChild[i]);
    }
    fread(&size,sizeof(int),1,file);
    float *_fit = new float[size];
    fread(_fit,sizeof(float),size,file);
    for(int i =0;i<size;i++){
      fits.push_back(_fit[i]);
    }
    float threshold;
    fread(&threshold,sizeof(float),1,file);
    thresholds.push_back(threshold);
  }
  fclose(file);
}
*/

vector<int> GAB::DetectFace(Mat img,vector<Rect>& rects,vector<float>& scores){
  const Options& opt = Options::GetInstance();
  int width = img.cols;
  int height = img.rows;
  for(int i = 0;i<29;i++){
    int win = pWinSize[i];
    if(win>width || win>height){
      break;
    }
    int step =(int) floor(win * 0.1);
    if(win>40)
      step = (int) floor(win*0.05);
    #pragma omp parallel for
    for(int y = 0;y<(height-win);y+=step){
      for(int x = 0;x<(width-win);x+=step){
        float score;
        Rect roi(x, y, win, win);
        Mat crop_img = img(roi).clone();
   //     cv::resize(img(roi), crop_img, Size(opt.objSize, opt.objSize));
        if(NPDClassify(img(roi),score,i)){
          #pragma omp critical
          {
          rects.push_back(roi);
          scores.push_back(score);
          }
        }
      }
    }
  }
  vector<int> picked;
  vector<int> Srect;
  picked = Nms(rects,scores,Srect,0.5,img);

  int imgWidth = img.cols;
  int imgHeight = img.rows;

  for(int i = 0;i<picked.size();i++){
    int idx = picked[i];
    int delta = floor(Srect[idx]*opt.enDelta);
    int y0 = max(rects[idx].y - floor(2.5 * delta),0);
    int y1 = min(rects[idx].y + Srect[idx] + floor(2.5 * delta),imgHeight);
    int x0 = max(rects[idx].x + delta,0);
    int x1 = min(rects[idx].x + Srect[idx] - delta,imgWidth);

    rects[idx].y = y0;
    rects[idx].x = x0;
    rects[idx].width = x1-x0 + 1;
    rects[idx].height = y1-y0 + 1;
  }
  
  return picked;
}

vector<int> GAB::Nms(vector<Rect>& rects, vector<float>& scores, vector<int>& Srect, float overlap, Mat Img) {
  int numCandidates = rects.size();
  Mat predicate = Mat::eye(numCandidates,numCandidates,IPL_DEPTH_1U);
  for(int i = 0;i<numCandidates;i++){
    for(int j = i+1;j<numCandidates;j++){
      int h = min(rects[i].y+rects[i].height,rects[j].y+rects[j].height) - max(rects[i].y,rects[j].y);
      int w = min(rects[i].x+rects[i].width,rects[j].x+rects[j].width) - max(rects[i].x,rects[j].x);
      int s = max(h,0)*max(w,0);

      if ((float)s/(float)(rects[i].width*rects[i].height+rects[j].width*rects[j].height-s)>=overlap){
        predicate.at<bool>(i,j) = 1;
        predicate.at<bool>(j,i) = 1;
      }
    }
  }

  vector<int> label;

  int numLabels = Partation(predicate,label);

  vector<Rect> Rects;
  Srect.resize(numLabels);
  vector<int> neighbors;
  neighbors.resize(numLabels);
  vector<float> Score;
  Score.resize(numLabels);

  for(int i = 0;i<numLabels;i++){
    vector<int> index;
    for(int j = 0;j<numCandidates;j++){
      if(label[j]==i)
        index.push_back(j);
    }
    vector<float> weight;
    weight = Logistic(scores,index);
    float sumScore=0;
    for(int j=0;j<weight.size();j++)
      sumScore+=weight[j];
    Score[i] = sumScore;
    neighbors[i]=index.size();

    if (sumScore == 0){
      for(int j=0;j<weight.size();j++)
        weight[j] = 1/sumScore;
    }
    else{
      for(int j=0;j<weight.size();j++)
        weight[j] = weight[j]/sumScore;
    }

    float size = 0;
    float col = 0;
    float row = 0;
    for(int j=0;j<index.size();j++){
      size += rects[index[j]].width*weight[j];
    }
    Srect[i] = (int)floor(size);
    for(int j=0;j<index.size();j++){
      col += (rects[index[j]].x + rects[index[j]].width/2)*weight[j];
      row += (rects[index[j]].y + rects[index[j]].width/2)*weight[j];
    }
    int x = floor(col-size/2);
    int y = floor(row-size/2);
    Rect roi(x,y,Srect[i],Srect[i]);
    Rects.push_back(roi);
  }


  predicate = Mat::zeros(numLabels,numLabels,IPL_DEPTH_1U);

  for(int i = 0;i<numLabels;i++){
    for(int j = i+1;j<numLabels;j++){
      int h = min(Rects[i].y+Rects[i].height,Rects[j].y+Rects[j].height) - max(Rects[i].y,Rects[j].y);
      int w = min(Rects[i].x+Rects[i].width,Rects[j].x+Rects[j].width) - max(Rects[i].x,Rects[j].x);
      int s = max(h,0)*max(w,0);

      if((float)s/(float)(Rects[i].width*Rects[i].height)>=overlap || (float)s/(float)(Rects[j].width*Rects[j].height)>=overlap)
      {
        predicate.at<bool>(i,j) = 1;
        predicate.at<bool>(j,i) = 1;
      }
    }
  }

  vector<int> flag;
  flag.resize(numLabels);
  for(int i = 0;i<numLabels;i++)
    flag[i]=1;

  for(int i = 0;i<numLabels;i++){
    vector<int> index;
    for(int j = 0;j<numLabels;j++){
      if(predicate.at<bool>(j,i)==1)
        index.push_back(j);
    }
    if(index.size()==0)
      continue;

    float s = 0;
    for(int j  = 0;j<index.size();j++){
      if(Score[index[j]]>s)
        s = Score[index[j]];
    }
    if(s>Score[i])
      flag[i]=0;
  }

  vector<int> picked;
  for(int i = 0;i<numLabels;i++){
    if(flag[i]){
      picked.push_back(i);
    }
  }

  int height = Img.rows;
  int width = Img.cols;

  for(int i = 0;i<picked.size();i++){
    int idx = picked[i];
    if(Rects[idx].x<0)
      Rects[idx].x = 0;

    if(Rects[idx].y<0)
      Rects[idx].y = 0;

    if(Rects[idx].y+Rects[idx].height>height)
      Rects[idx].height = height-Rects[idx].y;

    if(Rects[idx].x+Rects[idx].width>width)
      Rects[idx].width= width-Rects[idx].x;
  }

  rects = Rects;
  scores = Score;
  return picked;
}

vector<float> GAB::Logistic(vector<float> scores ,vector<int> index){
  vector<float> Y;
  for(int i = 0;i<index.size();i++){
    float tmp_Y = log(1+exp(scores[index[i]]));
    if(isinf(tmp_Y))
      tmp_Y = scores[index[i]];
    Y.push_back(tmp_Y);
  }
  return Y;
}

int GAB::Partation(Mat predicate,vector<int>& label){
  int N = predicate.cols;
  vector<int> parent;
  vector<int> rank;
  for(int i=0;i<N;i++){
    parent.push_back(i);
    rank.push_back(0);
  }

  for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
      if (predicate.at<bool>(i,j)==0)
        continue;
      int root_i = Find(parent,i);
      int root_j = Find(parent,j);

      if(root_j != root_i){
        if (rank[root_j] < rank[root_i])
          parent[root_j] = root_i;
        else if (rank[root_j] > rank[root_i])
          parent[root_i] = root_j;
        else{
          parent[root_j] = root_i;
          rank[root_i] = rank[root_i] + 1;
        }
      }
    }
  }

  int nGroups = 0;
  label.resize(N);
  for(int i=0;i<N;i++){
    if(parent[i]==i){
      label[i] = nGroups;
      nGroups++;
    }
    else label[i] = -1;
  }

  for(int i=0;i<N;i++){
    if(parent[i]==i)
      continue;
    int root_i = Find(parent,i);
    label[i]=label[root_i];
  }

  return nGroups;
}

int GAB::Find(vector<int>& parent,int x){
  int root = parent[x];
  if(root != x)
    root = Find(parent,root);
  return root;
}


Mat GAB::Draw(Mat& img, Rect& rects){
  Mat img_ = img.clone();
  rectangle(img_,rects,Scalar(0, 0, 255), 2);
  return img_;
}
