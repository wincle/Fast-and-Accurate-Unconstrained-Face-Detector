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
    /* \breif model path */
    string model_dir;
    /* \breif path of FDDB */
    string fddb_dir;
    /* \breif use for resize box */
    float enDelta;

  private:
    Options();
    Options(const Options& other);
    Options& operator=(const Options& other);

};
#endif // COMMON_HPP_
