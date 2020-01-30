# NPD

The C++ implementation of A Fast and Accurate Unconstrained Face Detector. 

The result is trained by 200k pos data and the template is 24*24, stages number is 620, model size is 540kb.

minFaceSize    |  speed(ms)  | cores 
:-----: | :----:    | :----:
80*80 |    30     | 1    
24x24 |    500    | 1
24*24 |   60     | 16

the detection result is test on FDDB data set (average 400*400)

# NOTICE

  The "1226model" is dump from matlab code which is from References, this model has 1226 stages.

  You must change the code in detection/LearnGAB.cpp:58-64. Because the difference between matlab and OpenCV. You should also change the coefficient in detection/LearnGAB.cpp:262-265 to fit the model.

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

```
hd.txt(Optional)
../data/hd/000001.jpg
../data/hd/000002.jpg
../data/hd/000003.jpg
...
```

the hd image is hard negative for init training , the size of it should to be the same with your model template(24 for me).

The config is in common.cpp 

# License

BSD 3-Clause

# Python Implemention

https://github.com/wincle/NPD_python

(It's much slow than C++ implementisn. PR is welcomed)

# References

http://www.cbsr.ia.ac.cn/users/scliao/projects/npdface/index.html
