#include "common.hpp"
#define MAX 100000000;

Options::Options(){ //default value
  objSize = 24;
  negRatio = 1.0;
  numFaces =  MAX;
  treeLevel = 8;
  maxNumWeaks = 1500;
  minDR = 1.0;
  maxFAR = 1e-16;
  minSamples = 10;
  faceDBFile = "../data/FaceDB.txt";
  nonfaceDBFile = "../data/NonfaceDB.txt";
  outFile = "../result";
  fddb_dir = "../data/fddb";
  tmpfile = "../data/tmpFaceDB.txt";
  minNegRatio = 0.2;
  trimFrac = 0.05;
  samFrac = 1.0;
  minLeafFrac = 0.01;
  minLeaf = 100;
  maxWeight = 100;
  augment = true;
  saveStep = 50;
}
