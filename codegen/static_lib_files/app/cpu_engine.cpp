/*
 * Copyright (c) 2017-2019 ARM Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "arm_compute/graph.h"
#include "support/ToolchainSupport.h"
#include "utils/CommonGraphOptions.h"
#include "utils/GraphUtils.h"
#include "utils/Utils.h"
#include <chrono>
#include <thread>
#include <string>
#include "cpu_engine.h"

using namespace arm_compute::utils;
using namespace arm_compute::graph::frontend;
using namespace arm_compute::graph_utils;

/** CONSTRUCTOR**/
//cpu_engine(int argc, char *argv[], float* input, float *output, Example *dnn_ptr, std::string name);
cpu_engine::cpu_engine(int argc, char *argv[], float* input, float *output, Example *dnn_ptr, std::string name){
   this->name = name;
   this->dnn_ptr = dnn_ptr;
   this->input = input;
   this->output = output;
   bool status = this->dnn_ptr->do_setup(argc, argv);
   if(!status)
        std::cerr << std::endl<< "CPU ENGINE "<<this->name<<" SETUP ERROR " << std::endl;
   else  
        std::cout<<"CPU ENGINE "<<this->name<<" CREATED!"<<std::endl;
}

/** DESTRUCTOR **/
cpu_engine::~cpu_engine(){

}

/** INFERENCE HERE **/
void cpu_engine::main(void *vpar) {
  try{  

        thread_info* par = (struct thread_info *) vpar;
  	setaffinity(par->core_id);        
        
        fifo_buf* in_buf_ptr  = par->get_fifo_buf_by_dst(this->name);
        fifo_buf* out_buf_ptr = par->get_fifo_buf_by_src(this->name);

        //in case Multi I/Os needed
        //std::vector<fifo_buf*> get_in_fifos(this->name);
        //std::vector<fifo_buf*> get_out_fifos(this->name);

        //max tokens port IP0
        int IP0_tokens = 0;
        int OP0_tokens = 0;

        //if(in_buf_ptr == nullptr) std::cout<<this->name<<" has null input buffer."<<std::endl;
        //else IP0_tokens = in_buf_ptr->in_rate;

        //if(out_buf_ptr == nullptr) std::cout<<this->name<<" has null output buffer."<<std::endl;
        //else OP0_tokens = out_buf_ptr->out_rate;
  	
       auto startTime = std::chrono::high_resolution_clock::now();
  	for(int img =0; img<frames; img++){
             //read
             if ( IP0_tokens > 0 )
               readSWF_CPU(in_buf_ptr->fifo, &input[0], IP0_tokens, in_buf_ptr->fifo_size);
                //std::cout<<"CPU ENGINE "<<this->name<<" reads "<<IP0_tokens<<" from input buffer of size "<<in_buf_ptr->fifo_size<<std::endl;

             //execute
             this->dnn_ptr->do_run();
            // std::cout<<"CPU ENGINE "<<this->name<<" executed! "<<std::endl;
 		
             //write
             if ( OP0_tokens > 0 )
	        writeSWF_CPU(out_buf_ptr->fifo, &output[0], OP0_tokens, out_buf_ptr->fifo_size);
               //std::cout<<"CPU ENGINE "<<this->name<<" writes "<<OP0_tokens<<" to output buffer of size "<<out_buf_ptr->fifo_size<<std::endl;
               
  	}
  	//auto endTime = std::chrono::high_resolution_clock::now();
  	//float totalTime = std::chrono::duration<float, std::milli>(endTime - startTime).count();
  	//std::cout<<this->name <<" (cpu core "<<par->core_id<<"): average over "<<frames<< " images = ~ "<<(totalTime/float(frames))<<" ms/img "<<std::endl;

 }

  catch(std::runtime_error &err){
   std::cerr << std::endl<< "CPU ENGINE ERROR " << err.what() << " " << (errno ? strerror(errno) : "") << std::endl;
   return;
 }
}
