#include <iostream>
#include "TrainDetector.hpp"
using namespace std;

/*! \breif command help */
static const char help[] = "NPD\n\n"
"train:  train a model ,if you already have, will resume it\n"
"test:   test FDDB data\n"
"live:   live demo with camera support\n";

/*!
 * \breif Command Dispatch
 */
int main(int argc, char* argv[]){
  TrainDetector dector;
  if (argc != 2) {
    printf(help);
  }
  else if (strcmp(argv[1], "train") == 0) {
    dector.Train();
  }
  else if (strcmp(argv[1], "test") == 0) {
    dector.FddbDetect();
  }
  else if (strcmp(argv[1], "live") == 0) {
    dector.Live();
  }
  return 0;
}
