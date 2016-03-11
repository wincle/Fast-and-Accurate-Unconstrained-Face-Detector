# NPD

The C++ implementation of NPD

Notice:The code is being Test!
I haven't train a model out with large size data.

# Difference with Matlab Code

- Neg Samples Mining in every Iter.
- Always training in LearGAB , don't jump out to TrainDector as Matlab Code.
- Can't retrain.

# How to use
- mkdir data

In data folder, you should creat two file named FaceDB.txt and NonFaceDB.txt.

```
FaceDB.txt
../data/face/00001.jpg x1 y1 x2 y2
../data/face/00002.jpg x1 y1 x2 y2
....
....
```

```
NonfaceDB.txt
../data/bg/000001.jpg
../data/bg/000002.jpg
../data/bg/000003.jpg
....
....
```

The config is in src/common.cpp 


#License

BSD 3-Clause

# References

http://www.cbsr.ia.ac.cn/users/scliao/projects/npdface/index.html
