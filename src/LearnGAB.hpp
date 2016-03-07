#include "common.hpp"
#include "LearnDQT.hpp"
#include <opencv2/core/core.hpp>

class GAB{
  public:
    void LearnGAB(cv::Mat faceFea, cv::Mat nonfaceFea);
    void save_cascade_to_file(vector<int>, vector<int>, vector<int>, vector< vector<unsigned char> >, vector<float>, float, float, int);
    int CalcTreeDepth(vector<int> leftChild, vector<int> rightChild, int node = 0);
    void TestDQT(float[], vector<float>, vector< vector<unsigned char> > , vector<int>, vector<int>, cv::Mat);
    void TestDQT(float[], vector<float>, vector< vector<unsigned char> > , vector<int>, vector<int>, cv::Mat, vector<int> );
    float TestSubTree(vector<float> ,vector< vector<unsigned char> > ,cv::Mat ,int ,int ,vector<int> leftChild, vector<int> rightChild,bool init);
    void CalcWeight(float F[], float Fx[], int y, int maxWeight, int);
    void CalcWeight(float F[], float Fx[], int y, int maxWeight, vector<int> negPassIndex);
    vector<DQT> trees;
  private:
};
