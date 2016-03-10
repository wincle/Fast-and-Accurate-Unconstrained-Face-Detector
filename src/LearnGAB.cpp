#include "LearnGAB.hpp"
#include <math.h>
#include <sys/time.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define max(a, b)  (((a) > (b)) ? (a) : (b))
#define min(a, b)  (((a) < (b)) ? (a) : (b))

using namespace cv;

GAB::GAB(int size){
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

void GAB::LearnGAB(DataSet& pos, DataSet& neg){
  const Options& opt = Options::GetInstance();
  timeval start, end;
  timeval Tstart, Tend;
  int nPos = pos.size;
  int nNeg = neg.size;

  Mat faceFea = pos.Extract();
  pos.ImgClear();
  printf("Extract pos feature finish\n");
  Mat nonfaceFea = neg.Extract();
  printf("Extract neg feature finish\n");

  float _FAR=1.0;
  int nFea=0;
  float aveEval=0;

  float *w = new float[nPos];
  for (int t = stages;t<opt.maxNumWeaks;t++){
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
    float Ttime = (Tend.tv_sec - Tstart.tv_sec)*1000+(Tend.tv_usec - Tstart.tv_usec)/1000;

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

    TestDQT(pos.Fx,fit,cutpoint,leftChild,rightChild,posX);
    TestDQT(neg.Fx,fit,cutpoint,leftChild,rightChild,negX);
    

    vector<int> negPassIndex;
    for(int i=0; i<nNegSam; i++)
      negPassIndex.push_back(i);

    memcpy(w,pos.Fx,nPos*sizeof(float));
    sort(w,w+nPos);
    int index = max(floor(nPos*(1-opt.minDR)),1);
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
    float time = (end.tv_sec - start.tv_sec)*1000+(end.tv_usec - start.tv_usec)/1000;

    int nNegPass = negPassIndex.size();
    printf("FAR(t)=%.2f%%, FAR=%.2g, depth=%d, nFea(t)=%d, nFea=%d, cost=%.3f.\n",far*100.,_FAR,depth,feaId.size(),nFea,mincost);
    printf("\t\tnNegPass=%d, aveEval=%.3f, alltime=%.3fms, Ttime = %.3fms, meanT=%.3fms.\n", nNegPass, aveEval, time, Ttime, time/(t-stages+1));

    
    if(_FAR<=opt.maxFAR){
      printf("\n\nThe training is converged at iteration %d. FAR = %.2f%%\n", t, _FAR * 100);
      break;
    }


    SaveIter(feaId,leftChild,rightChild,cutpoint,fit,threshold,far,depth);

    gettimeofday(&Tstart,NULL); 

    neg.Remove(negPassIndex);
    MiningNeg(negPassIndex.size(),neg);

    nonfaceFea = neg.Extract();
    pos.CalcWeight(1,opt.maxWeight);
    neg.CalcWeight(-1,opt.maxWeight);
    
    gettimeofday(&Tend,NULL);
    Ttime = (Tend.tv_sec - Tstart.tv_sec)*1000+(Tend.tv_usec - Tstart.tv_usec)/1000;
    printf("update weight time:%.3fms\n",Ttime);

    printf("posFx\n");
    for(int i = 0;i<nPos;i++)
      printf("%f ",pos.Fx[i]);
    printf("\n");

    printf("negFx\n");
    for(int i = 0;i<nPos;i++)
      printf("%f ",pos.Fx[i]);
    printf("\n");

    printf("posW\n");
    for(int i = 0;i<nPos;i++)
      printf("%f ",pos.W[i]);
    printf("\n");

    printf("negW\n");
    for(int i = 0;i<nNeg;i++)
      printf("%f ",neg.W[i]);
    printf("\n");
  }


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
    score[i] = TestSubTree(fit,cutpoint,x,0,i,leftChild,rightChild);

  for(int i =0;i<n;i++)
    posFx[i]+=score[i];
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

bool GAB::NPDClassify(Mat test,float &score){
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

void GAB::MiningNeg(int st,DataSet& neg){
  const Options& opt = Options::GetInstance();
  int pool_size = omp_get_max_threads();
  vector<Mat> region_pool(pool_size);
  int n = neg.size;
  srand(time(0));

  while(st<n){

    int current_idx = rand()%(neg.list.size());
    neg.img = imread(neg.list[current_idx], CV_LOAD_IMAGE_GRAYSCALE);
    #pragma omp parallel for
    for(int i = 0;i<pool_size;i++){
      region_pool[i] = neg.NextImage(i);
    }
    
    for (int i = 0; i < pool_size; i++) {
      float score = 0;
      if(NPDClassify(region_pool[i].clone(),score)){
        neg.imgs.push_back(region_pool[i].clone());
        neg.Fx[st]=score;
        st++;
      }
    }
  }
}
