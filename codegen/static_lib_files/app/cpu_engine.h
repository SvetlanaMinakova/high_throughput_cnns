#ifndef cpu_engine_H
#define cpu_engine_H

#include "arm_compute/graph.h"
#include "support/ToolchainSupport.h"
#include "utils/CommonGraphOptions.h"
#include "utils/GraphUtils.h"
#include "utils/Utils.h"
#include <chrono>
#include <thread>
#include <string>
#include "types.h"
#include "fifo.h"

using namespace arm_compute::utils;
using namespace arm_compute::graph::frontend;
using namespace arm_compute::graph_utils;

class cpu_engine{
public:
//gpu_engine(gpu_partition* dnn_ptr, float* input, float *output, cudaStream_t* stream_ptr, std::string name);
 cpu_engine(int argc, char *argv[], float* input, float *output, Example *dnn_ptr, std::string name);
 ~cpu_engine();
  void main(void *par);

  Example* dnn_ptr;
  float *input;
  float *output;
  std::string name; 

};
#endif // cpu_engine_H
