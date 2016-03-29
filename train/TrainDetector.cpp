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
