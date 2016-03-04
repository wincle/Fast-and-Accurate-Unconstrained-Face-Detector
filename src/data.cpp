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
  x = y = 0;
  current_idx = 0;
  transform_type = ORIGIN;
  random_shuffle(list.begin(),list.end());
  MoreNeg(ceil(size*opt.negRatio));
}
void DataSet::MoreNeg(const int n){
  const Options& opt = Options::GetInstance();
  int pool_size = omp_get_max_threads();
  imgs.resize(size+pool_size+1);
  vector<Mat> region_pool(pool_size);
  int num = 0;
  while(num<n){
    for(int i = 0;i<pool_size;i++)
      region_pool[i] = NextImage();
    for (int i = 0; i < pool_size; i++) {
      imgs[num]=region_pool[i].clone();
      num ++;
    }
  }
}

Mat DataSet::NextImage() {
  const Options& opt = Options::GetInstance();
  const int w = opt.objSize;
  const int h = opt.objSize;

  NextState();

  Mat region(opt.objSize,opt.objSize,CV_8UC1);
  Rect roi(x, y, w, h);
  Mat img = imread(list[current_idx], CV_LOAD_IMAGE_GRAYSCALE);
  region = img(roi).clone();

  switch (transform_type) {
    case ORIGIN:
      break;
    case ORIGIN_R:
      flip(region, region, 0);
      transpose(region, region);
      break;
    case ORIGIN_RR:
      flip(region, region, -1);
      break;
    case ORIGIN_RRR:
      flip(region, region, 1);
      transpose(region, region);
      break;
    case ORIGIN_FLIP:
      flip(region, region, 1);
      break;
    case ORIGIN_FLIP_R:
      flip(region, region, -1);
      transpose(region, region);
      break;
    case ORIGIN_FLIP_RR:
      flip(region, region, -1);
      flip(region, region, 1);
      break;
    case ORIGIN_FLIP_RRR:
      flip(region, region, 0);
      transpose(region, region);
      flip(region, region, 1);
      break;
    default:
      printf("Unsupported Transform of Negative Sample\n");
      break;
  }
  return region;
}

void DataSet::NextState() {
  const Options& opt = Options::GetInstance();
  const double scale_factor = 0.8;
  const int w = opt.objSize;
  const int h = opt.objSize;
  const int x_step = w/4;
  const int y_step = w/4;
  Mat img = imread(list[current_idx], CV_LOAD_IMAGE_GRAYSCALE);

  const int width = img.cols;
  const int height = img.rows;

  switch (transform_type) {
    case ORIGIN:
      transform_type = ORIGIN_R;
      return;
    case ORIGIN_R:
      transform_type = ORIGIN_RR;
      return;
    case ORIGIN_RR:
      transform_type = ORIGIN_RRR;
      return;
    case ORIGIN_RRR:
      transform_type = ORIGIN_FLIP;
      return;
    case ORIGIN_FLIP:
      transform_type = ORIGIN_FLIP_R;
      return;
    case ORIGIN_FLIP_R:
      transform_type = ORIGIN_FLIP_RR;
      return;
    case ORIGIN_FLIP_RR:
      transform_type = ORIGIN_FLIP_RRR;
      return;
    case ORIGIN_FLIP_RRR:
      transform_type = ORIGIN;
      break;
    default:
      printf("Unsupported Transform of Negative Sample\n");
      break;
  }

  x += x_step; // move x
  if (x + w >= width) {
    x = 0;
    y += y_step; // move y
    if (y + h >= height) {
      x = y = 0;
      int width_ = int(img.cols * scale_factor);
      int height_ = int(img.rows * scale_factor);
      cv::resize(img, img, Size(width_, height_)); // scale image
      if (img.cols < w || img.rows < h) {
        // next image
        while (true) {
          current_idx++; // next image
          if (current_idx >= list.size()) {
            // Add background image list online
            printf("Run out of Negative Samples! :-(\n");
            continue;
          }
          //LOG("Use %d th Nega Image %s", current_idx + 1, list[current_idx].c_str());
          img = cv::imread(list[current_idx], 0);
          if (!img.data || img.cols <= w || img.rows <= h) {
            if (!img.data) {
              //LOG("Can not open image %s, Skip it", list[current_idx].c_str());
            }
            else {
              //LOG("Image %s is too small, Skip it", list[current_idx].c_str());
            }
          }
          else {
            // successfully get another background image
            break;
          }
        }
      }
    }
  }
}



