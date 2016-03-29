# NPD

The C++ implementation of NPD

The result is trained by 200k pos data and the template is 24*24, stages number is 620, model size is 540kb.


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

The config is in src/common.cpp 

#TODO

Speed Up the Detection

#Contact with me

You can join the Google group NPD_C_Group to talk with each other.

https://groups.google.com/forum/?hl=en#!forum/npd_c_group

#License

BSD 3-Clause

# References

http://www.cbsr.ia.ac.cn/users/scliao/projects/npdface/index.html
