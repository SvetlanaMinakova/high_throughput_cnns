#include <iostream>
#include <map>
#include <vector>
#include <thread>
#include <chrono>
#include "cuda_runtime_api.h"
#include "gpu_partition.h"
#include "gpu_engine.h"
#include "Subnet2.h"
#include "arm_compute/graph.h"
#include "cpu_engine.h"
#include "Subnet0.h"
#include "Subnet1.h"

using namespace std;

int main (int argc, char **argv) {
  cudaDeviceReset();
  std::cout<<"***DNN building phase.***"<<std::endl;
  ////////////////////////////////////////////////////////////
  // CREATE PARTITIONS (DNN-DEPENDENT TOPOLOGY INIT/CLEAN) //
  std::cout<<" - partitions creation."<<std::endl;
  //GPU
  Subnet2 p2;
  
  //CPU
  Subnet0 p0;
  Subnet1 p1;
  
  /////////////////////////////////////////////////////////////
  // CREATE DNN/DNN PARTITIONS I/O BUFFERS //
  std::cout<<" - I/O buffers allocation."<<std::endl;
  //GPU
  float p2_input[p2.batchSize *  p2.INPUT_H * p2.INPUT_W * p2.INPUT_C] = {0};
  float p2_output[p2.batchSize * p2.OUTPUT_SIZE] = {0};
  
  //CPU
  float p0_input[p0.batchSize *  p0.INPUT_H * p0.INPUT_W * p0.INPUT_C] = {0};
  float p0_output[p0.batchSize * p0.OUTPUT_SIZE] = {0};
  float p1_input[p1.batchSize *  p1.INPUT_H * p1.INPUT_W * p1.INPUT_C] = {0};
  float p1_output[p1.batchSize * p1.OUTPUT_SIZE] = {0};
  
  // CREATE ENGINES (OBJECTS TO RUN DNN PARTITIONS) //
  std::cout<<" - Engines creation."<<std::endl;
  //GPU
  cudaStream_t p2_stream; 
  CHECK(cudaStreamCreate(&p2_stream));
  gpu_engine e2 (&p2, p2_input, p2_output, &p2_stream, "e2");
  
  //CPU
  cpu_engine e0 (argc, argv, p0_input, p0_output, &p0, "e0");
  cpu_engine e1 (argc, argv, p1_input, p1_output, &p1, "e1");
  //CPU engine pointers
  std::vector<cpu_engine*> cpu_engine_ptrs;
  cpu_engine_ptrs.push_back(&e0);
  cpu_engine_ptrs.push_back(&e1);
  
  //GPU engine pointers
  std::vector<gpu_engine*> gpu_engine_ptrs;
  gpu_engine_ptrs.push_back(&e2);
  
  /////////////////////////////////////////////////////////////
  //  PTHREAD thread_infoparams //
  std::cout<<" - Pthread info-params creation."<<std::endl;
  
  int subnets = 3;
  int core_ids[subnets] = {0, 1, 5};
  //Allocate memory for pthread_create() arguments
  const int num_threads = subnets;
  struct thread_info *thread_info = (struct thread_info*)(calloc(num_threads, sizeof(struct thread_info)));
  
  //  allocate CPU cores
  for(int i = 0;i<num_threads; i++)
    thread_info[i].core_id = core_ids[i];
 
 /////////////////////////////////////////////////////////////
 // INFERENCE //
 std::cout<<"*** DNN inference phase.***"<<std::endl;
 
 std::cout<<" - Threads creation and execution."<<std::endl;
 
 auto startTime = std::chrono::high_resolution_clock::now();
 
 //Create and run posix threads
 std::thread my_thread0(&cpu_engine::main, &e0, &thread_info[0]);//(CPU)
 std::thread my_thread1(&cpu_engine::main, &e1, &thread_info[1]);//(CPU)
 std::thread my_thread2(&gpu_engine::main, &e2, &thread_info[2]);//(GPU)
 
 //join posix threads
 my_thread0.join();
 my_thread1.join();
 my_thread2.join();
 
 auto endTime = std::chrono::high_resolution_clock::now();
 float totalTime = std::chrono::duration<float, std::milli>(endTime - startTime).count();
 std::cout<<"Average over "<<frames<< " images = ~ "<<(totalTime/float(frames))<<" ms/img"<<std::endl;
 
 /////////////////////////////////////////////////////////////
 // CLEAN MEMORY //
 std::cout<<"*** DNN destruction phase ***"<<std::endl;
 
 //Destroy GPU streams
 std::cout<<" - CUDA streams destruction"<<std::endl;
 cudaStreamDestroy(p2_stream);
 
 //Destroy CPU partitions
 std::cout<<" - CPU partitions destruction"<<std::endl;
 p0.do_teardown();
 p1.do_teardown();
 
 //delete pthread parameters
 std::cout<<" - Pthread parameters destruction"<<std::endl;
 free(thread_info);
 
 
 return 0;
 }
