#include "data.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <omp.h>

using namespace cv;


DataSet::DataSet(){
}

void DataSet::LoadDataSet(DataSet& pos, DataSet& neg){
  const Options& opt = Options::GetInstance();
  printf("Loading Pos data\n");
  pos.LoadPositiveDataSet(opt.faceDBFile);
  printf("Pos data finish\n");
  printf("Loading Neg data\n");
  neg.LoadNegativeDataSet(opt.nonfaceDBFile,pos.size);
  printf("Neg data finish\n");
}
void DataSet::LoadPositiveDataSet(const string& positive){
  const Options& opt = Options::GetInstance();
  FILE* file = fopen(positive.c_str(), "r");
  char buff[300];
  vector<string> path;
  vector<Rect> bboxes;
  imgs.clear();
  while (fscanf(file, "%s", buff) > 0) {
    path.push_back(string(buff));
    Rect bbox;
    fscanf(file, "%d%d%d%d", &bbox.x, &bbox.y, &bbox.width, &bbox.height);    
    bboxes.push_back(bbox);
    }
  fclose(file);
  const int n = path.size();
  size = n;
  imgs.resize(size);
  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    Mat origin = imread(path[i], CV_LOAD_IMAGE_GRAYSCALE);
    if (!origin.data) {
      printf("Can not open %s",path[i].c_str());
    }
    Mat face = origin(bboxes[i]);
    Mat img;
    cv::resize(face, img, Size(opt.objSize, opt.objSize));
    imgs[i] = img;
  }
  random_shuffle(imgs.begin(),imgs.end());
  is_pos=true;
}

void DataSet::LoadNegativeDataSet(const string& negative, int pos_num){
  const Options& opt = Options::GetInstance();
  FILE* file = fopen(negative.c_str(), "r");
  char buff[256];
  list.clear();
  while (fscanf(file, "%s", buff) > 0) {
    list.push_back(buff);
  }
  size = pos_num;
  imgs.resize(size);
  current_idx = 0;
  random_shuffle(list.begin(),list.end());
  MoreNeg(ceil(size*opt.negRatio));
}
void DataSet::MoreNeg(const int n){
  const Options& opt = Options::GetInstance();
  int pool_size = omp_get_max_threads();
  imgs.resize(size+pool_size);
  vector<Mat> region_pool(pool_size);
  int num = 0;
  srand(time(0));
  while(num<n){
    current_idx = rand()%list.size();
    img = imread(list[current_idx], CV_LOAD_IMAGE_GRAYSCALE);
//    #pragma omp parallel for
    for(int i = 0;i<pool_size;i++){
      region_pool[i] = NextImage(i);
    }
    for (int i = 0; i < pool_size; i++) {
      imgs[num]=region_pool[i].clone();
      num ++;
    }
  }
}

Mat DataSet::NextImage(int i) {
  const Options& opt = Options::GetInstance();
  const int w = opt.objSize;
  const int h = opt.objSize;


  srand(time(0)*(i+1));

  const int width = img.cols;
  const int height = img.rows;
  int x=0,y=0,s=0;

  s = w+rand()%(min(width,height)-w);
  x = rand()%(width-s);
  y = rand()%(height-s);

  Rect roi(x, y, s, s);

  Mat crop_img = img(roi);
  Mat region;
  resize(crop_img,region,Size(w,h));

  return region;
}
