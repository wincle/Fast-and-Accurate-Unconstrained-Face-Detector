#include "TrainDetector.hpp"
#include "LearnGAB.hpp"
#include <sys/time.h>
#include <omp.h>

using namespace cv;

void TrainDetector::Train(){
  Options& opt = Options::GetInstance();
  DataSet pos,neg;
  DataSet::LoadDataSet(pos, neg);

  GAB Gab(pos.size);
  Gab.LearnGAB(pos,neg);
//  Gab.Save();
//  pos.Clear();
//  neg.Clear();
}
