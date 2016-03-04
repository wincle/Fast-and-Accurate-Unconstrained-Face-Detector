CC = g++
OPENCV = -L/opt/meituan/local/lib/ -lopencv_highgui -lopencv_core -lopencv_imgproc -I/opt/meituan/local/include/opencv
all : Train
Train: common.cpp common.hpp TrainDetector_Demo.cpp TrainDetector.cpp TrainDetector.hpp data.hpp data.cpp LearnGAB.cpp LearnGAB.hpp LearnDQT.cpp LearnDQT.hpp
	$(CC) common.cpp data.cpp TrainDetector_Demo.cpp TrainDetector.cpp LearnGAB.cpp LearnDQT.cpp $(OPENCV) -fopenmp -o Train -g
