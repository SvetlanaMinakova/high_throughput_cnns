CXX=aarch64-linux-gnu-g++

CXXFLAGS= -std=c++11 -Wl,--allow-shlib-undefined

ARM_PATH=/home/nvidia/arm_cl/ComputeLibrary

ARM_INCFLAG = -I${ARM_PATH} -I${ARM_PATH}/include

ARM_SOURCES = ${ARM_PATH}/utils/Utils.cpp ${ARM_PATH}/utils/GraphUtils.cpp ${ARM_PATH}/utils/CommonGraphOptions.cpp
CXXLIB = -L${ARM_PATH} -larm_compute_graph -larm_compute -larm_compute_core -lpthread

CUDA_FLAGS= -Wall -std=c++11 -O2

CUDA_PATH=/usr/local/cuda-9.0

CUDA_INCFLAG = -I"/usr/local/cuda-9.0/include" -I"/usr/local/include" -I"../include" -I"../common" -I"/usr/local/cuda-9.0/include" -I"../../include"  -D_REENTRANT

CUDA_LIB = -L"" -L"/usr/local/cuda-9.0/targets/x86_64-linux/lib64" -L"/usr/local/lib" -L"../lib" -L"/usr/local/cuda-9.0/lib64" -L"/usr/local/cuda-9.0/lib64" -L"../../lib"   -L./ -Wl,--start-group -lnvinfer -lnvparsers -lnvinfer_plugin -lcudnn -lcublas -lcudart_static -lnvToolsExt -lcudart -lrt -ldl -lpthread -Wl,--end-group


OBJS = Subnet0.o Subnet1.o cpu_engine.o Utils.o GraphUtils.o CommonGraphOptions.o gpu_engine.o Subnet2.o gpu_partition.o fifo.o appMain.o

PRG = appMain

all: ${PRG}

appMain: ${OBJS}
	${CXX} -o appMain ${OBJS} $(ARM_INCFLAG) $(CUDA_INCFLAG) ${CXXLIB} ${CUDA_LIB} ${CXXFLAGS}

Utils.o: 
	${CXX} ${ARM_PATH}/utils/Utils.cpp $(ARM_INCFLAG) ${CXXLIB} ${CXXFLAGS} -c -g $?

GraphUtils.o: 
	${CXX} ${ARM_PATH}/utils/GraphUtils.cpp $(ARM_INCFLAG) ${CXXLIB} ${CXXFLAGS} -c -g $?

CommonGraphOptions.o: 
	${CXX} ${ARM_PATH}/utils/CommonGraphOptions.cpp $(ARM_INCFLAG) ${CXXLIB} ${CXXFLAGS} -c -g $?

cpu_engine.o: 
	${CXX} cpu_engine.cpp $(ARM_INCFLAG) ${CXXLIB} ${CXXFLAGS} -c -g $?

Subnet0.o: 
	${CXX} Subnet0.cpp $(ARM_INCFLAG) ${CXXLIB} ${CXXFLAGS} -c -g $?

Subnet1.o: 
	${CXX} Subnet1.cpp $(ARM_INCFLAG) ${CXXLIB} ${CXXFLAGS} -c -g $?

gpu_partition.o: 
	${CXX} gpu_partition.cpp $(CUDA_INCFLAG) ${CUDA_LIB} ${CUDA_FLAGS} -c -g $?

gpu_engine.o: 
	${CXX} gpu_engine.cpp $(CUDA_INCFLAG) ${CUDA_LIB} ${CUDA_FLAGS} -c -g $?

Subnet2.o: 
	${CXX} Subnet2.cpp $(CUDA_INCFLAG) ${CUDA_LIB} ${CUDA_FLAGS} -c -g $?

fifo.o: fifo.cpp
	${CXX} ${CXXFLAGS} -c -g $?

appMain.o: appMain.cpp
	${CXX} appMain.cpp $(ARM_INCFLAG) $(CUDA_INCFLAG) ${CXXLIB} ${CUDA_LIB} ${CXXFLAGS} -c -g $?



clean:
	rm -rf *~ *.o 
