#include "common.hpp"
#define MAX 100000000;

Options::Options(){ //default value
  objSize = 20;
  negRatio = 1.0;
  finalNegs = 100;
  numFaces =  MAX;
  numThreads = 24;
  treeLevel = 8;
  maxNumWeaks = 1226;
  minDR = 1.0;
  maxFAR = 1e-16;
  minSamples = 100;
  faceDBFile = "../data/FaceDB.txt";
  nonfaceDBFile = "../data/NonfaceDB.txt";
  outFile = "../result.txt";
  treeLevel = 4; 
  maxNumWeaks = 1000;
  minDR = 1.0;
  maxFAR = 1e-16;
  minSamples = 10;
  minNegRatio = 0.2;
  trimFrac = 0.05;
  samFrac = 1.0;
  minLeafFrac = 0.01;
  minLeaf = 100;
  maxWeight = 100;
  numThreads = 24;
}
