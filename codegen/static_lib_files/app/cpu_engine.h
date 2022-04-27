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
#include "Subnet.h"

using namespace arm_compute::utils;
using namespace arm_compute::graph::frontend;
using namespace arm_compute::graph_utils;

class cpu_engine: public Subnet{
public:
   // constructor
   cpu_engine(int argc, char *argv[], Example *dnn_ptr, std::string name): Subnet (name, frames, 0, 0){
   this->name = name;
   this->dnn_ptr = dnn_ptr;
   bool status = this->dnn_ptr->do_setup(argc, argv);
   if(!status)
        std::cerr << std::endl<< "CPU ENGINE "<<this->name<<" SETUP ERROR " << std::endl;
   else  
        std::cout<<"CPU ENGINE "<<this->name<<" CREATED!"<<std::endl;
   }
 
   //desctructor
   ~cpu_engine();
   void main(void *thread_par) override;
   Example* dnn_ptr;
   std::string name; 
};
#endif // cpu_engine_H
