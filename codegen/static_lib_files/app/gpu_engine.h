#ifndef gpu_engine_H
#define gpu_engine_H

#include "common.h"
#include "cuda_runtime_api.h"
#include <map>
#include <vector>
#include <thread>
#include "types.h"
#include "gpu_partition.h"
#include "Subnet.h"

using namespace std;

class gpu_engine: public Subnet{
public:
// constructor and destructor
    gpu_engine(gpu_partition* dnn_ptr, cudaStream_t* stream_ptr, std::string name): Subnet (name, frames, 0, 0){
    this->dnn_ptr = dnn_ptr;
    this->cuda_stream_ptr = stream_ptr;
    this->name = name;

    bool setup_err = false;

    if(dnn_ptr==nullptr){
        std::cerr << std::endl<< "GPU ENGINE "<<this->name<<" SETUP ERROR: DNN PARTITION PTR IS NULL" << std::endl;
        setup_err=true;
   }
   
   if(!setup_err)  
        std::cout<<"GPU ENGINE "<<this->name<<" CREATED!"<<std::endl;
 }
 
 ~gpu_engine();
 
  void main(void* thread_par) override; 
  gpu_partition* dnn_ptr;
  cudaStream_t* cuda_stream_ptr;
  std::string name; 
};
#endif // gpu_engine_H
