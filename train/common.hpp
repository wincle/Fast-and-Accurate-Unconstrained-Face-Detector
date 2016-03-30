#ifndef COMMON_HPP_
#define COMMON_HPP_
#include <string>
#include <stdio.h>
using namespace std;

/*
 * \breif Configure of NPD
 */
class Options{
  public:
    static inline Options& GetInstance() {
      static Options opt;
      return opt;
    }
    /* \breif Size of Template */
    int objSize;
    /* \breif a text file for positive dataset */
    string faceDBFile;
    /* \breif a text file for negative dataset */
    string nonfaceDBFile;
    /* \breif path of model */
    string outFile;
    /* \breif path of FDDB */
    string fddb_dir;
    /* \breif a text file for resume training status */
    string tmpfile;
    /* \breif Init Neg Samples */
    string initNeg;
    /* \breif depth of a stage */
    int treeLevel;
    /* \breif max number of stages */
    int maxNumWeaks;
    /* \breif threads to use */
    int numThreads;
    /* \breif recall of positive in every stages */
    double minDR;
    /* \breif end condition of the training */
    double maxFAR;
    /* \breif max value of weight */
    int maxWeight;
    /* \breif factor for decide leaf number */
    double minLeafFrac;
    /* \breif minimum leaf number */
    int minLeaf;
    /* \breif factor to decide how many samples should be filter befor training a stage */
    double trimFrac;
    /* \breif minimum samples required */
    int minSamples;
    /* \breif data augment or not */
    bool augment;
    /* \breif step of stages to save the model */
    int saveStep;
    /* \breif generate init neg if need */
    bool generate_hd;
    /* \breif use for resize box */
    float enDelta;
    /* \use hd or not */
    bool useInitHard;
    /* \Ration of neg/pos */
    float negRatio;

  private:
    Options();
    Options(const Options& other);
    Options& operator=(const Options& other);

};
#endif // COMMON_HPP_
