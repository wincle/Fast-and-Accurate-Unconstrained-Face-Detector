#include "TrainDetector.hpp"
#include "LearnGAB.hpp"
#include <omp.h>

using namespace cv;

TrainDetector::TrainDetector(){
}

void TrainDetector::Train(){
  Options& opt = Options::GetInstance();
  DataSet pos,neg;
  DataSet::LoadDataSet(pos, neg);
  numFaces = pos.size;
  numNegs = ceil(numFaces * opt.negRatio);
  Mat faceFea = Extract(pos);
  printf("Extract pos feature finish\n");

  int trainTime = 0;
  int numStages = 0;

  while (true){
    Mat NonFaceFea = Extract(neg);
    printf("Extract neg feature finish\n");
    GAB Gab;
    Gab.LearnGAB(faceFea,NonFaceFea);




    break;



  }

}

Mat TrainDetector::Extract(DataSet data){
  Options& opt = Options::GetInstance();
  int numProcs = omp_get_num_procs();
  int numThreads = (int) floor(numProcs * 0.8);    
  omp_set_num_threads(numThreads);

  Mat ppNpdTable = Mat(256,256,CV_8UC1);


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

  size_t height = opt.objSize;
  size_t width = opt.objSize;
  size_t numPixels = height * width;
  size_t numImgs = data.size;
  size_t feaDims = numPixels * (numPixels - 1) / 2;

  Mat fea = Mat(feaDims,numImgs,CV_8UC1);
  int x1,y1,x2,y2,d;
  for(int k = 0; k < numImgs; k++)
  {
    d = 0;
    printf("extract %d imgs\n",k);
    Mat img = data.imgs[k];
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
