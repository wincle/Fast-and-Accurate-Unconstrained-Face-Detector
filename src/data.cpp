#include "data.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <omp.h>

using namespace cv;


DataSet::DataSet(){
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
  x = 0;
  y = 0;
  current_id = 0;

}

void DataSet::LoadDataSet(DataSet& pos, DataSet& neg, int stages){
  const Options& opt = Options::GetInstance();
  printf("Loading Pos data\n");
  pos.LoadPositiveDataSet(opt.faceDBFile,stages);
  printf("Pos data finish %d %d\n",pos.size,pos.imgs.size());
  printf("Loading Neg data\n");
  neg.LoadNegativeDataSet(opt.nonfaceDBFile,pos.size,stages);
  printf("Neg data finish %d %d\n",neg.size,neg.imgs.size());
}
void DataSet::LoadPositiveDataSet(const string& positive,int stages){
  const Options& opt = Options::GetInstance();
  FILE *file;
  vector<int> is_flip;
  char buff[300];
  vector<string> path;
  vector<Rect> bboxes;

  if(!opt.augment || stages == 0){
    file = fopen(positive.c_str(), "r");
    while (fscanf(file, "%s", buff) > 0) {
      path.push_back(string(buff));
      Rect bbox;
      fscanf(file, "%d%d%d%d", &bbox.x, &bbox.y, &bbox.width, &bbox.height);  
      bboxes.push_back(bbox);
      is_flip.push_back(0);
    }
  }
  else{
    file = fopen(opt.tmpfile.c_str(), "r");
    while (fscanf(file, "%s", buff) > 0) {
      path.push_back(string(buff));
      Rect bbox;
      int flip;
      fscanf(file, "%d%d%d%d%d", &bbox.x, &bbox.y, &bbox.width, &bbox.height, &flip);  
      is_flip.push_back(flip);
      bboxes.push_back(bbox);
    }
  }

  fclose(file);
  const int n = path.size();
  size = n;
  if(!opt.augment || stages != 0){
    imgs.resize(size);
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
      Mat origin = imread(path[i], CV_LOAD_IMAGE_GRAYSCALE);
      if (!origin.data) {
        printf("Can not open %s",path[i].c_str());
      }
      Mat tmp = origin.clone();
      if(is_flip[i])
        flip(tmp,tmp,1);
      Mat face = tmp(bboxes[i]);
      Mat img;
      cv::resize(face, img, Size(opt.objSize, opt.objSize));
      imgs[i] = img.clone();
    }
  }
  else
  {
    FILE* tmpfile = fopen(opt.tmpfile.c_str(),"w");
    size = size*10;
    imgs.resize(size);
    #pragma omp parallel for
    for(int i = 0; i < n; i++) {
      Mat origin = imread(path[i], CV_LOAD_IMAGE_GRAYSCALE);
      if (!origin.data) {
        printf("Can not open %s",path[i].c_str());
      }
      int type = 0;
      int stmp,xtmp,ytmp;
      int s = bboxes[i].width;

      while(type < 5){
        srand(time(0)+type);
        do{
          stmp = s*((float)(rand()%20)/100.0+0.9);
          xtmp = bboxes[i].x+s*((float)(rand()%10)/100.0-0.05);
          ytmp = bboxes[i].y+s*((float)(rand()%10)/100.0-0.05);
        }while(xtmp<0 || ytmp<0 || (xtmp+stmp)>origin.cols || (ytmp+stmp)>origin.rows);
        Rect rect(xtmp,ytmp,stmp,stmp);
        fprintf(tmpfile,"%s %d %d %d %d 0\n",path[i].c_str(),xtmp,ytmp,stmp,stmp);

        Mat face = origin(rect);
        Mat img;
        cv::resize(face, img, Size(opt.objSize, opt.objSize));
        imgs[i*10+type]=img.clone();
        type ++;
      }

      flip(origin,origin,1);
      bboxes[i].x = origin.cols - bboxes[i].x - bboxes[i].width;

      while(type < 10){
        srand(time(0)+type);
        do{
          stmp = s*((float)(rand()%20)/100.0+0.9);
          xtmp = bboxes[i].x+s*((float)(rand()%10)/100.0-0.05);
          ytmp = bboxes[i].y+s*((float)(rand()%10)/100.0-0.05);
        }while(xtmp<0 || ytmp<0 || (xtmp+stmp)>origin.cols || (ytmp+stmp)>origin.rows);
        Rect rect(xtmp,ytmp,stmp,stmp);
        fprintf(tmpfile,"%s %d %d %d %d 1\n",path[i].c_str(),xtmp,ytmp,stmp,stmp);

        Mat face = origin(rect);
        Mat img;
        cv::resize(face, img, Size(opt.objSize, opt.objSize));
        imgs[i*10+type]=img.clone();
        type ++;
      }
    }
    fclose(tmpfile);
  }
  random_shuffle(imgs.begin(),imgs.end());
  initWeights();
}

void DataSet::LoadNegativeDataSet(const string& negative, int pos_num, int stages){
  const Options& opt = Options::GetInstance();
  FILE* file = fopen(negative.c_str(), "r");
  char buff[256];
  list.clear();
  while (fscanf(file, "%s", buff) > 0) {
    list.push_back(buff);
  }
  size = pos_num;
  random_shuffle(list.begin(),list.end());
  imgs.reserve(size + omp_get_max_threads());
  initWeights();
  for(int k = 0; k<omp_get_max_threads() ; k++){
    Mat img = imread(list[k],CV_LOAD_IMAGE_GRAYSCALE);
    NegImgs.push_back(img.clone());
  }
  if(stages == 0)
    MoreNeg(ceil(size*opt.negRatio));
}
void DataSet::MoreNeg(const int n){
  const Options& opt = Options::GetInstance();
  int pool_size = omp_get_max_threads();
  vector<Mat> region_pool(pool_size);
  int num = 0;

  int all = 0;

  while(num<n){
    #pragma omp parallel for
    for(int i = 0;i<pool_size;i++){
      region_pool[i] = NextImage(i);
    }
    for (int i = 0; i < pool_size; i++) {
      imgs.push_back(region_pool[i].clone());
      num ++;
    }
  }
}

Mat DataSet::NextImage(int i) {
  const Options& opt = Options::GetInstance();
  const int w = opt.objSize;

  Mat img = NegImgs[i];

  const int width = img.cols;
  const int height = img.rows;
  int x=0,y=0,s=0;
  int type=0;

  srand(time(0)+i);

  s = w+(int)(rand()%(min(width,height)-w));
  x = rand()%(width-s);
  y = rand()%(height-s);
  type=rand()%(8);

  Rect roi(x, y, s, s);

  Mat crop_img = img(roi);
  Mat region;
  resize(crop_img,region,Size(w,w));

  switch(type){
    case 0:
      break;
    case 1:
      flip(region, region, 0);
      transpose(region, region);
      break;
    case 2:
      flip(region, region, -1);
      break;
    case 3:
      flip(region, region, 1);
      transpose(region, region);
      break;
    case 4:
      flip(region, region, 1);
      break;
    case 5:
      flip(region, region, -1);
      transpose(region, region);
      break;
    case 6:
      flip(region, region, -1);
      flip(region, region, 1);
      break;
    case 7:
      flip(region, region, 0);
      transpose(region, region);
      flip(region, region, 1);
      break;
    default:
      printf("error type!\n");
      break;
  }

  return region;
}

void DataSet::ImgClear(){
  imgs.clear();
  vector<Mat> blank;
  imgs.swap(blank);
}

void DataSet::Remove(vector<int> PassIndex){
  int passNum = PassIndex.size();

  vector<Mat> tmpImgs;
  float* tmpFx = new float[size+omp_get_max_threads()];
  for(int i = 0;i<size+omp_get_max_threads();i++)
    tmpFx[i] = 0;

  for(int i = 0;i<passNum;i++){
    tmpImgs.push_back(imgs[PassIndex[i]]);
    tmpFx[i] = Fx[PassIndex[i]];
  }

  memcpy(Fx,tmpFx,(size+omp_get_max_threads())*sizeof(float));
  imgs = tmpImgs;
  delete []tmpFx;
}

Mat DataSet::Extract(){
  Options& opt = Options::GetInstance();
  int numThreads = omp_get_num_procs();
  omp_set_num_threads(numThreads);

  size_t height = opt.objSize;
  size_t width = opt.objSize;
  size_t numPixels = height * width;
  size_t numImgs = size;
  size_t feaDims = numPixels * (numPixels - 1) / 2;

  Mat fea = Mat(feaDims,numImgs,CV_8UC1);


  #pragma omp parallel for
  for(int k = 0; k < numImgs; k++)
  {
    int x1,y1,x2,y2,d;
    d = 0;
    Mat img = imgs[k];
    for(int i = 0; i < numPixels; i++)
    {
      y1 = i%opt.objSize;
      x1 = i/opt.objSize;
      for(int j = i+1; j < numPixels; j ++)
      {
        y2 = j%opt.objSize;
        x2 = j/opt.objSize;

        fea.at<uchar>(d++,k) = ppNpdTable.at<uchar>(img.at<uchar>(x1,y1),img.at<uchar>(x2,y2));
      }
    }
  }
  return fea;
}

void DataSet::initWeights(){
  W = new float[size];
  for(int i = 0;i<size;i++)
    W[i]=1./size;
  Fx = new float[size+omp_get_max_threads()];
  for(int i = 0;i<size;i++)
    Fx[i]=0;
}

void DataSet::CalcWeight(int y, int maxWeight){
  float s = 0;
  for(int i = 0;i<size;i++){
    W[i]=min(exp(-y*Fx[i]),float(maxWeight));
    s += W[i];
  }
  if (s == 0)
    for(int i = 0;i<size;i++)
      W[i]=1/size;
  else
    for(int i = 0;i<size;i++)
      W[i]/=s;
}

void DataSet::Clear(){
  delete []W;
  delete []Fx;
}
