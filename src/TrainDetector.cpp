#include "TrainDetector.hpp"
#include "LearnGAB.hpp"
#include <sys/time.h>
#include <omp.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

void TrainDetector::Train(){
  Options& opt = Options::GetInstance();
  DataSet pos,neg;

  GAB Gab;
  Gab.LoadModel(opt.outFile);
  DataSet::LoadDataSet(pos, neg, Gab.stages);
  Gab.LearnGAB(pos,neg);
  Gab.Save();
  pos.Clear();
  neg.Clear();
}

void TrainDetector::FddbDetect(){
  Options& opt = Options::GetInstance();

  const char* fddb_dir=opt.fddb_dir.c_str();
  string prefix = opt.fddb_dir + string("/");
  GAB Gab;
  Gab.LoadModel(opt.outFile);

  #pragma omp parallel for
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
      index = Gab.DetectFace(img,rects,scores);
      printf("%s\n%d\n",path,index.size());
      fprintf(fout,"%s\n%d\n",path,index.size());
      for(int i = 0;i < index.size(); i++){
        printf("%d %d %d %d %lf\n", rects[index[i]].x, rects[index[i]].y, rects[index[i]].width, rects[index[i]].height, scores[index[i]]);
        fprintf(fout, "%d %d %d %d %lf\n", rects[index[i]].x, rects[index[i]].y, rects[index[i]].width, rects[index[i]].height, scores[index[i]]);
      }
    }
    fclose(fin);
    fclose(fout);
  }
}
