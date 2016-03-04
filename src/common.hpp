#ifndef COMMON_HPP_
#define COMMON_HPP_
#include <string>
using namespace std;


class Options{
  public:
    static inline Options& GetInstance() {
      static Options opt;
      return opt;
    }
    int objSize;
    double negRatio;
    int finalNegs;
    int numFaces;
    int numThreads;
    string faceDBFile;
    string nonfaceDBFile;
    string outFile;
    int treeLevel;
    int maxNumWeaks;
    double minDR;
    double maxFAR;
    int minSamples;
    int maxWeight;
    double minNegRatio;
    double minLeafFrac;
    int minLeaf;
    double samFrac;
    double trimFrac;

  private:
    Options();
    Options(const Options& other);
    Options& operator=(const Options& other);

};
#endif // COMMON_HPP_

