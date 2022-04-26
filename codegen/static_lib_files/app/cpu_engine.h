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

using namespace arm_compute::utils;
using namespace arm_compute::graph::frontend;
using namespace arm_compute::graph_utils;

class cpu_engine{
public:
 cpu_engine(int argc, char *argv[], Example *dnn_ptr, std::string name); // float* input, float *output parameters removed
 ~cpu_engine();
  void main(void *par);

  Example* dnn_ptr;
  
  /**
  // I/O buffers moved into SharedBuffer objects
  float *input;
  float *output;
  */
  std::string name; 

};
#endif // cpu_engine_H
