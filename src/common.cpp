#include "common.hpp"

Options::Options(){ //default value
  objSize = 24;
  treeLevel = 8;
  maxNumWeaks = 1000;
  minDR = 1.0;
  maxFAR = 0;
  minSamples = 100000;
  faceDBFile = "../data/FaceDB.txt";
  nonfaceDBFile = "../data/NonfaceDB.txt";
  outFile = "../result";
  fddb_dir = "../data/fddb";
  tmpfile = "../data/tmpFaceDB.txt";
  trimFrac = 0.05;
  minLeafFrac = 0.01;
  minLeaf = 100;
  maxWeight = 100;
  augment = true;
  saveStep = 10;
}
