# NPD

The C++ implementation of NPD

Notice:The code is being Test!

The result is trained by 200k pos data and the template is 24*24 , the stages num is 100

# Difference with Matlab Code

- Neg Samples Mining in every Iter.
- Always training in LearGAB , don't jump out to TrainDector as Matlab Code.

# How to use
- you should mkdir data first

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

#Contact with me

You can join the Google group NPD_C_Group to talk with each other.
https://groups.google.com/forum/#!forum/npd_c_group

#License

BSD 3-Clause

# References

http://www.cbsr.ia.ac.cn/users/scliao/projects/npdface/index.html
