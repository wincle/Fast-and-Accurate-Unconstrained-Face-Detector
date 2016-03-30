#include "common.hpp"

Options::Options(){ //default value
  objSize = 24;
  treeLevel = 8;
  maxNumWeaks = 1000;
  numThreads = 16;
  minDR = 1.0;
  maxFAR = 0;
  faceDBFile = "../data/FaceDB.txt";
  nonfaceDBFile = "../data/NonfaceDB.txt";
  outFile = "../result";
  fddb_dir = "../data/fddb";
  tmpfile = "../data/tmpFaceDB.txt";
  initNeg = "../data/hd.txt";
  trimFrac = 0.005;
  minLeafFrac = 0.01;
  minLeaf = 100;
  minSamples = 20000;
  maxWeight = 100;
  augment = true;
  saveStep = 10;
  generate_hd = false;
  enDelta = 0.1;
  useInitHard = false;
  negRatio = 0.5;
}
