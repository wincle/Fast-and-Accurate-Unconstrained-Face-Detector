#include "LearnDQT.hpp"

DQT::DQT(){
  const Options& opt = Options::GetInstance();

  ppNpdTable = cv::Mat(256,256,CV_8UC1);
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

  size_t numPixels = opt.objSize*opt.objSize;
  for(int i = 0; i < numPixels; i++)
  {
    for(int j = i+1; j < numPixels; j ++)
    {
      lpoints.push_back(i);
      rpoints.push_back(j);
    }
  }
}

float DQT::Learn(cv::Mat posX,cv::Mat negX, float pPosW[], float pNegW[], vector<int> posIndex,vector<int> negIndex, int minLeaf, vector<int> &feaId, vector<int> &leftChild, vector<int> &rightChild, vector< vector<unsigned char> > &cutpoint, vector<float> &fit){
  const Options& opt = Options::GetInstance();
  int treeLevel = opt.treeLevel;
  int numThreads = opt.numThreads;
  int nTotalPos = posX.cols;
  int nTotalNeg = negX.cols;
  int nPos = posIndex.size();
  int nNeg = negIndex.size();
  int numPixels = opt.objSize*opt.objSize;

  vector<unsigned char *> ppPosX(numPixels);
  ppPosX[0] = (unsigned char *)posX.data;
  for(int i = 1; i < numPixels; i++) ppPosX[i] = ppPosX[i-1] + nTotalPos;

  vector<unsigned char *> ppNegX(numPixels);
  ppNegX[0] = (unsigned char *)negX.data;
  for(int i = 1; i < numPixels; i++) ppNegX[i] = ppNegX[i-1] + nTotalNeg;

  int pPosIndex[nPos];
  int pNegIndex[nNeg];

  for(int i =0;i<nPos;i++)
    pPosIndex[i]=posIndex[i];
  for(int i =0;i<nNeg;i++)
    pNegIndex[i]=negIndex[i];

  float minCost = LearnDQT(ppPosX, ppNegX, pPosW, pNegW, pPosIndex, pNegIndex, nPos, nNeg, treeLevel, minLeaf, numThreads, 0, feaId, cutpoint, leftChild, rightChild, fit);

  return minCost;

}

float DQT::LearnDQT(vector<unsigned char *> &posX, vector<unsigned char *> &negX, float *posW, float *negW, int *posIndex, int *negIndex, int nPos, int nNeg, int treeLevel, int minLeaf, int numThreads, float parentFit, vector<int> &feaId, vector< vector<unsigned char> > &cutpoint, vector<int> &leftChild, vector<int> &rightChild, vector<float> &fit)
{
  int _feaId;
  unsigned char _cutpoint[2];
  float _fit[2];

  float minCost = LearnQuadStump(posX, negX, posW, negW, posIndex, negIndex, nPos, nNeg, minLeaf, numThreads, parentFit,  _feaId, _cutpoint, _fit);

  if(_feaId < 0) return minCost;

  feaId.push_back(_feaId);
  cutpoint.push_back(vector<unsigned char>(_cutpoint, _cutpoint+2));
  leftChild.push_back(-1);
  rightChild.push_back(-2);

  if(treeLevel <= 1)
  {
    fit.push_back(_fit[0]);
    fit.push_back(_fit[1]);        
    return minCost;
  }

  int nPos1 = 0, nNeg1 = 0, nPos2 = 0, nNeg2 = 0;
  vector<int> posIndex1(nPos), posIndex2(nPos), negIndex1(nNeg), negIndex2(nNeg);

  for(int j = 0; j < nPos; j++)
  {
    int x,y;
    GetPoints(_feaId,&x,&y);
    unsigned char Fea = ppNpdTable.at<uchar>(posX[size_t(x)][size_t(posIndex[j])],posX[size_t(y)][size_t(posIndex[j])]);

    if(Fea < _cutpoint[0] || Fea > _cutpoint[1])
    {
      posIndex1[nPos1++] = posIndex[j];
    }
    else
    {
      posIndex2[nPos2++] = posIndex[j];
    }
  }

  for(int j = 0; j < nNeg; j++)
  {
    int x,y;
    GetPoints(_feaId,&x,&y);
    unsigned char Fea = ppNpdTable.at<uchar>(negX[size_t(x)][size_t(negIndex[j])],negX[size_t(y)][size_t(negIndex[j])]);

    if(Fea < _cutpoint[0] || Fea > _cutpoint[1])
    {
      negIndex1[nNeg1++] = negIndex[j];
    }                                                  
    else
    {
      negIndex2[nNeg2++] = negIndex[j];
    }
  }

  vector<int> feaId1, feaId2, leftChild1, leftChild2, rightChild1, rightChild2;
  vector< vector<unsigned char> > cutpoint1, cutpoint2;
  vector<float> fit1, fit2;

  float minCost1 = LearnDQT(posX, negX, posW, negW, &posIndex1[0], &negIndex1[0], nPos1, nNeg1, treeLevel - 1, minLeaf, numThreads, _fit[0], feaId1, cutpoint1, leftChild1, rightChild1, fit1);

  float minCost2 = LearnDQT(posX, negX, posW, negW, &posIndex2[0], &negIndex2[0], nPos2, nNeg2, treeLevel - 1, minLeaf, numThreads, _fit[1], feaId2, cutpoint2, leftChild2, rightChild2, fit2);

  if(feaId1.empty() && feaId2.empty())
  {
    fit.push_back(_fit[0]);
    fit.push_back(_fit[1]);        
    return minCost;
  }

  if(minCost1 + minCost2 >= minCost)
  {
    fit.push_back(_fit[0]);
    fit.push_back(_fit[1]);
    return minCost;
  }

  minCost = minCost1 + minCost2;

  if(feaId1.empty())
  {
    fit.push_back(_fit[0]);
  }
  else
  {
    feaId.insert(feaId.end(), feaId1.begin(), feaId1.end());
    cutpoint.insert(cutpoint.end(), cutpoint1.begin(), cutpoint1.end());
    fit = fit1;

    for(int i = 0; i < leftChild1.size(); i++)
    {
      if(leftChild1[i] >= 0) leftChild1[i]++;
      if(rightChild1[i] >= 0) rightChild1[i]++;
    }

    leftChild[0] = 1;
    leftChild.insert(leftChild.end(), leftChild1.begin(), leftChild1.end());
    rightChild.insert(rightChild.end(), rightChild1.begin(), rightChild1.end());
  }

  int numBranchNodes = (int) feaId.size();
  int numLeafNodes = (int) fit.size();

  if(feaId2.empty())
  {
    fit.push_back(_fit[1]);
    rightChild[0] = -(numLeafNodes + 1);
  }
  else
  {
    feaId.insert(feaId.end(), feaId2.begin(), feaId2.end());
    cutpoint.insert(cutpoint.end(), cutpoint2.begin(), cutpoint2.end());
    fit.insert(fit.end(), fit2.begin(), fit2.end());

    for(int i = 0; i < leftChild2.size(); i++)
    {
      if(leftChild2[i] >= 0) leftChild2[i] += numBranchNodes;
      else leftChild2[i] -= numLeafNodes;

      if(rightChild2[i] >= 0) rightChild2[i] += numBranchNodes;
      else rightChild2[i] -= numLeafNodes;
    }

    leftChild.insert(leftChild.end(), leftChild2.begin(), leftChild2.end());
    rightChild[0] = numBranchNodes;
    rightChild.insert(rightChild.end(), rightChild2.begin(), rightChild2.end());
  }

  return minCost;
}


float DQT::LearnQuadStump(vector<unsigned char *> &posX, vector<unsigned char *> &negX, float *posW, float *negW, int *posIndex, int *negIndex, int nPos, int nNeg, int minLeaf, int numThreads, float parentFit, int &feaId, unsigned char (&cutpoint)[2], float (&fit)[2])
{
  float w = 0;
  for(int i = 0; i < nPos; i++) w += posW[ posIndex[i] ];
  float minCost = w * (parentFit - 1) * (parentFit - 1);

  w = 0;
  for(int i = 0; i < nNeg; i++) w += negW[ negIndex[i] ];
  minCost += w * (parentFit + 1) * (parentFit + 1);

  feaId = -1;
  if(nPos == 0 || nNeg == 0 || nPos + nNeg < 2 * minLeaf) return minCost;

  int feaDims = (int) posX.size()*(posX.size()-1)/2;
  minCost = 1e16f;

  omp_set_num_threads(numThreads);
   
  // process each dimension  
  #pragma omp parallel for
  for(int i = 0; i < feaDims; i++)
  {
    int count[256];
    float posWHist[256];
    float negWHist[256];

    memset(count, 0, 256 * sizeof(int));

    int x,y;
    GetPoints(i,&x,&y);

    WeightHist(posX[x], posX[y], posW, posIndex, nPos, count, posWHist);
    WeightHist(negX[x], negX[y], negW, negIndex, nNeg, count, negWHist);

    float posWSum = 0;
    float negWSum = 0;

    for(int bin = 0; bin < 256; bin++)
    {
      posWSum += posWHist[bin];
      negWSum += negWHist[bin];
    }        

    int totalCount = nPos + nNeg;
    float wSum = posWSum + negWSum;

    float minMSE = 3.4e38f;
    int thr0 = -1, thr1;
    float fit0, fit1;

    for(int v = 0; v < 256; v++) // lower threshold
    {
      int rightCount = 0;
      float rightPosW = 0;
      float rightNegW = 0;

      for(int u = v; u < 256; u++) // upper threshold
      {
        rightCount += count[u];
        rightPosW += posWHist[u];
        rightNegW += negWHist[u];

        if(rightCount < minLeaf) continue;

        int leftCount = totalCount - rightCount;
        if(leftCount < minLeaf) break;                

        float leftPosW = posWSum - rightPosW;
        float leftNegW = negWSum - rightNegW;

        float leftFit, rightFit;

        if(leftPosW + leftNegW <= 0) leftFit = 0.0f;
        else leftFit = (leftPosW - leftNegW) / (leftPosW + leftNegW);

        if(rightPosW + rightNegW <= 0) rightFit = 0.0f;
        else rightFit = (rightPosW - rightNegW) / (rightPosW + rightNegW);

        float leftMSE = leftPosW * (leftFit - 1) * (leftFit - 1) + leftNegW * (leftFit + 1) * (leftFit + 1);
        float rightMSE = rightPosW * (rightFit - 1) * (rightFit - 1) + rightNegW * (rightFit + 1) * (rightFit + 1);

        float mse = leftMSE + rightMSE;

        if(mse < minMSE)
        {
          minMSE = mse;
          thr0 = v;
          thr1 = u;
          fit0 = leftFit;
          fit1 = rightFit;
        }
      }
    }
    if(thr0 == -1) continue;

    if(minMSE < minCost)
    {
      #pragma omp critical // modify the record by a single thread
      {
        minCost = minMSE;
        feaId = i;
        cutpoint[0] = (unsigned char) thr0;
        cutpoint[1] = (unsigned char) thr1;
        fit[0] = fit0;
        fit[1] = fit1;
      }
    }
  } // end omp parallel for feaDims
  return minCost;
}


void DQT::WeightHist(unsigned char *X, unsigned char *Y, float *W, int *index, int n, int count[256], float wHist[256])
{
  memset(wHist, 0, 256 * sizeof(float));

  for(int j = 0; j < n; j++)
  {
    unsigned char bin = ppNpdTable.at<uchar>(X[ index[j] ],Y[ index[j] ]);
    count[bin]++; 
    wHist[bin] += W[ index[j] ];
  }
} 

void DQT::GetPoints(int feaid, int *x, int *y){
  const Options& opt = Options::GetInstance();
  int lpoint = lpoints[feaid];
  int rpoint = rpoints[feaid];
  *x = lpoint;
  *y = rpoint;
}
