#include "LearnGAB.hpp"
#include <math.h>
#include <sys/time.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <map>

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

  minRate = 0.001;

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
      if(NPDClassify(pos.imgs[i].clone(),score)){
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
    printf("start training %d stages \n",t);
    gettimeofday(&start,NULL);

    nNeg = neg.size;
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
    if(neg.size<opt.minSamples){
      MiningNeg(nPos,neg);
    }

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

  feaIds.push_back(feaId); 
  leftChilds.push_back(leftChild); 
  rightChilds.push_back(rightChild);
  cutpoints.push_back(cutpoint);
  fits.push_back(fit);
  thresholds.push_back(threshold);
  stages++;

}
void GAB::Save(){
  const Options& opt = Options::GetInstance();
  FILE* file;
  file = fopen(opt.outFile.c_str(), "wb");

  fwrite(&opt.objSize,sizeof(int),1,file);
  fwrite(&stages,sizeof(int),1,file);
  int size;
  for(int i = 0;i<stages;i++){
    size = feaIds[i].size();
    int *feaId = new int[size];
    for(int j = 0;j<size;j++)
      feaId[j] = feaIds[i][j];
    fwrite(&size,sizeof(int),1,file);
    fwrite(feaId,sizeof(int),feaIds[i].size(),file);
    delete []feaId;
    unsigned char* cutpoint = new unsigned char[2*size];
    for(int j = 0;j<size;j++){
      cutpoint[2*j] = cutpoints[i][j][0];
      cutpoint[2*j+1] = cutpoints[i][j][1];
    }
    fwrite(cutpoint,sizeof(unsigned char),2*cutpoints[i].size(),file);
    delete []cutpoint;
    int *leftChild = new int[size];
    for(int j = 0;j<size;j++){
      leftChild[j] = leftChilds[i][j];
    }
    fwrite(leftChild,sizeof(int),leftChilds[i].size(),file);
    delete []leftChild;
    int *rightChild = new int[size];
    for(int j = 0;j<size;j++){
      rightChild[j] = rightChilds[i][j];
    }
    fwrite(rightChild,sizeof(int),rightChilds[i].size(),file);
    delete []rightChild;
    size = fits[i].size();
    float *fit = new float[size];
    for(int j = 0;j<size;j++){
      fit[j] = fits[i][j];
    }
    fwrite(&size,sizeof(int),1,file);
    fwrite(fit,sizeof(float),fits[i].size(),file);
    delete []fit;
    float threshold = thresholds[i];
    fwrite(&threshold,sizeof(float),1,file);

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

void GAB::GetPoints(int feaid, int *x, int *y){
  const Options& opt = Options::GetInstance();
  int lpoint = lpoints[feaid];
  int rpoint = rpoints[feaid];
  *x = lpoint;
  *y = rpoint;
}

void GAB::MiningNeg(int n,DataSet& neg){
  const Options& opt = Options::GetInstance();
  int pool_size = omp_get_max_threads();
  vector<Mat> region_pool(pool_size);
  int st = neg.size;
  int all = 0;
  int iter_all = 0;
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
      if(NPDClassify(region_pool[i].clone(),score)){
        #pragma omp critical 
        {
          neg.imgs.push_back(region_pool[i].clone());
          neg.Fx[st]=score;
          st++;
        }
      }
      iter_all++;
      all++;
    }
    rate = ((double)(need))/(double)iter_all;
    if(rate < minRate){
      need = n - st;
      iter_all = 0;
      neg.current_id += pool_size;
      if ((neg.current_id+pool_size)>=neg.list.size()){
        neg.current_id = 0;
        minRate /= 10;
      }
      for(int k =0 ;k< pool_size;k++){
        Mat img = imread(neg.list[k+neg.current_id],CV_LOAD_IMAGE_GRAYSCALE);
        neg.NegImgs[k] = img.clone();
      }
    }
  }
  neg.size = n;
  printf("mining success rate %lf\n",rate);
}

void GAB::LoadModel(string path){
  FILE* file;
  if((file = fopen(path.c_str(), "rb"))==NULL)
    return;
  int size;

  fread(&DetectSize,sizeof(int),1,file);
  fread(&stages,sizeof(int),1,file);
  printf("stages num :%d\n",stages);
  for(int j = 0;j<stages;j++){
    printf("stage: %d\n",j);
    vector<int> feaId, leftChild, rightChild;
    vector< vector<unsigned char> > cutpoint;
    vector<float> fit;
    float threshold;

    fread(&size,sizeof(int),1,file);
    int *_feaId = new int[size];
    fread(_feaId,sizeof(int),size,file);
    for(int i = 0;i<size;i++){
      feaId.push_back(_feaId[i]);
    }
    printf("\n");
    unsigned char *_cutpoint = new unsigned char[size*2];
    fread(_cutpoint,sizeof(unsigned char),2*size,file);
    for(int i =0;i<size;i++){
      vector<unsigned char> cut;
      cut.push_back(_cutpoint[2*i]);
      cut.push_back(_cutpoint[2*i+1]);
      cutpoint.push_back(cut);
    }
    int *_leftChild = new int[size];
    fread(_leftChild,sizeof(int),size,file);
    for(int i =0;i<size;i++)
      leftChild.push_back(_leftChild[i]);
    int *_rightChild = new int[size];
    fread(_rightChild,sizeof(int),size,file);
    for(int i =0;i<size;i++){
      rightChild.push_back(_rightChild[i]);
    }
    fread(&size,sizeof(int),1,file);
    float *_fit = new float[size];
    fread(_fit,sizeof(float),size,file);
    for(int i =0;i<size;i++){
      fit.push_back(_fit[i]);
    }
    fread(&threshold,sizeof(float),1,file);

    feaIds.push_back(feaId);
    leftChilds.push_back(leftChild);
    rightChilds.push_back(rightChild);
    cutpoints.push_back(cutpoint);
    fits.push_back(fit);
    thresholds.push_back(threshold);

    delete []_feaId;
    delete []_cutpoint;
    delete []_leftChild;
    delete []_rightChild;
    delete []_fit;
  }
  fclose(file);
}

vector<int> GAB::DetectFace(Mat img,vector<Rect>& rects,vector<float>& scores){
  const Options& opt = Options::GetInstance();
  int width = img.cols;
  int height = img.rows;
  int win = opt.objSize;
  float factor = 1.2;
  int x = 0;
  int y = 0;
  Mat crop_img;
  while(win <= width && win <= height){
    int step = win * 0.1;
    while (y<=(height-win)){
      while (x<=(width-win)) {
        float score;
        Rect roi(x, y, win, win);
        cv::resize(img(roi), crop_img, Size(opt.objSize, opt.objSize));
        if(NPDClassify(crop_img,score)){
          rects.push_back(roi);
          scores.push_back(score);
        }
        x += step;
      }
      x = 0;
      y += step;
    }
    x = 0;
    y = 0;
    win = win*factor;
  }
  vector<int> picked;
  picked = Nms(rects,scores,0.3);
  return picked;
}


vector<int> GAB::Nms(vector<Rect>& rects, vector<float>& scores, float overlap) {
  const int n = rects.size();
  vector<float> areas(n);

  typedef std::multimap<float, int> ScoreMapper;
  ScoreMapper map;
  for (int i = 0; i < n; i++) {
    map.insert(ScoreMapper::value_type(scores[i], i));
    areas[i] = rects[i].width*rects[i].height;
  }

  int picked_n = 0;
  vector<int> picked(n);
  while (map.size() != 0) {
    int last = map.rbegin()->second; // get the index of maximum score value
    picked[picked_n] = last;
    picked_n++;
    for (ScoreMapper::iterator it = map.begin(); it != map.end();) {
      int idx = it->second;
      float x1 = max(rects[idx].x, rects[last].x);
      float y1 = max(rects[idx].y, rects[last].y);
      float x2 = min(rects[idx].x + rects[idx].width, rects[last].x + rects[last].width);
      float y2 = min(rects[idx].y + rects[idx].height, rects[last].y + rects[last].height);
      float w = max(0., x2 - x1);
      float h = max(0., y2 - y1);
      float ov = w*h / (areas[idx] + areas[last] - w*h);
      if (ov > overlap) {
        ScoreMapper::iterator tmp = it;
        tmp++;
        map.erase(it);
        it = tmp;
      }
      else{
        it++;
      }
    }
  }

  picked.resize(picked_n);
  return picked;
}

