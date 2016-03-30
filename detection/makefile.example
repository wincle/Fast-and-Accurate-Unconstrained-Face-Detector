CC = g++
OPENCV = -L/local/lib/ -lopencv_highgui -lopencv_core -lopencv_imgproc -I/local/include/opencv
all : demo
demo: common.cpp common.hpp TrainDetector.cpp TrainDetector.hpp LearnGAB.cpp LearnGAB.hpp
	$(CC) demo.cpp common.cpp TrainDetector.cpp LearnGAB.cpp $(OPENCV) -fopenmp -o demo -O3 -DNDEBUG
