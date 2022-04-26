#ifndef gpu_engine_H
#define gpu_engine_H

#include "common.h"
#include "cuda_runtime_api.h"
#include <map>
#include <vector>
#include <thread>
#include "types.h"
#include "fifo.h"
#include "gpu_partition.h"

using namespace std;

class gpu_engine{
public:
 gpu_engine(gpu_partition* dnn_ptr, float* input, float *output, cudaStream_t* stream_ptr, std::string name);
 ~gpu_engine();
  void main(void* par); 

  gpu_partition* dnn_ptr;
  float *input;
  float *output;
  cudaStream_t* cuda_stream_ptr;
  std::string name; 
};
#endif // gpu_engine_H
