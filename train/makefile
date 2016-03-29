CC = g++
OPENCV = -L/opt/meituan/local/lib/ -lopencv_highgui -lopencv_core -lopencv_imgproc -I/opt/meituan/local/include/opencv
all : demo
demo: common.cpp common.hpp TrainDetector.cpp TrainDetector.hpp data.hpp data.cpp LearnGAB.cpp LearnGAB.hpp LearnDQT.cpp LearnDQT.hpp
	$(CC) demo.cpp common.cpp data.cpp TrainDetector.cpp LearnGAB.cpp LearnDQT.cpp $(OPENCV) -fopenmp -o demo -O3	-DNDEBUG
