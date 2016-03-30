#include "TrainDetector.hpp"
#include "LearnGAB.hpp"
#include <sys/time.h>
#include <omp.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

void TrainDetector::Detect(){
  Options& opt = Options::GetInstance();

  GAB Gab;
  Gab.LoadModel(opt.outFile);

  timeval start, end;
  float time = 0;

  string path = "1.jpg";
  Mat img = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
  vector<Rect> rects;
  vector<float> scores;
  vector<int> index;
  gettimeofday(&start,NULL);
  index = Gab.DetectFace(img,rects,scores);
  gettimeofday(&end,NULL);
  float t = 1000 * (end.tv_sec-start.tv_sec)+ (end.tv_usec-start.tv_usec)/1000;
  printf("use time:%f\n",t);
  for(int i = 0;i < index.size(); i++){
    printf("%d %d %d %d %lf\n", rects[index[i]].x, rects[index[i]].y, rects[index[i]].width, rects[index[i]].height, scores[index[i]]);
   for (int i = 0; i < index.size(); i++) {
    if(scores[index[i]]>0)
      img = Gab.Draw(img, rects[index[i]]);
  }
  imwrite("2.jpg",img);
 }
}


void TrainDetector::FddbDetect(){
  Options& opt = Options::GetInstance();

  const char* fddb_dir=opt.fddb_dir.c_str();
  string prefix = opt.fddb_dir + string("/");
  GAB Gab;
  Gab.LoadModel(opt.outFile);

  timeval start, end;
  float time = 0;

  for(int i = 1;i<=10;i++){
    char fddb[300];
    char fddb_out[300];
    sprintf(fddb, "%s/FDDB-folds/FDDB-fold-%02d.txt", fddb_dir, i);
    sprintf(fddb_out, "%s/result/fold-%02d-out.txt", fddb_dir, i);
    FILE* fin = fopen(fddb, "r");
    FILE* fout = fopen(fddb_out, "w");
    char path[300];

    while (fscanf(fin, "%s", path) > 0) {
      string full_path = prefix + string(path) + string(".jpg");
      Mat img = imread(full_path, CV_LOAD_IMAGE_GRAYSCALE);
      vector<Rect> rects;
      vector<float> scores;
      vector<int> index;
      gettimeofday(&start,NULL);
      index = Gab.DetectFace(img,rects,scores);
      gettimeofday(&end,NULL);
      float t = 1000 * (end.tv_sec-start.tv_sec)+ (end.tv_usec-start.tv_usec)/1000;
      time += t;
      printf("%s\n%d\n",path,index.size());
      printf("use time:%f\n",t);
      fprintf(fout,"%s\n%d\n",path,index.size());
      for(int i = 0;i < index.size(); i++){
        printf("%d %d %d %d %lf\n", rects[index[i]].x, rects[index[i]].y, rects[index[i]].width, rects[index[i]].height, scores[index[i]]);
        fprintf(fout, "%d %d %d %d %lf\n", rects[index[i]].x, rects[index[i]].y, rects[index[i]].width, rects[index[i]].height, scores[index[i]]);
      }
    }
    printf("all time:%f\n",time);
    fclose(fin);
    fclose(fout);
  }
}

void TrainDetector::Live() {
  Options& opt = Options::GetInstance();

  VideoCapture cap(0);
  if (!cap.isOpened()) {
    printf("Can not open Camera, Please Check it!");
    return;
  }

  GAB Gab;
  Gab.LoadModel(opt.outFile);

  while (true) {
    Mat frame;
    Mat gray;
    cap >> frame;

    cvtColor(frame, gray, CV_BGR2GRAY);
    vector<Rect> rects;
    vector<float> scores;
    vector<int> index;
    index = Gab.DetectFace(gray,rects,scores);

    for (int i = 0; i < index.size(); i++) {
      if(scores[index[i]]>100)
        frame = Gab.Draw(frame, rects[index[i]]);
    }
    cv::imshow("live", frame);
    int key = cv::waitKey(30);
    if (key == 27) {
      break;
    }
  }
}
