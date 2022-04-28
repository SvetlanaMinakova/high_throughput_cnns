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

/** DESTRUCTOR **/
cpu_engine::~cpu_engine(){

}

/** INFERENCE HERE **/
void cpu_engine::main(void *thread_par) {
    try{
        // assign CPU core to current thread
        auto* par = (struct ThreadInfo*) thread_par;
        setaffinity(par->core_id);  

        for (int i=0; i<frames;i++){
            // wait until input data is ready for reading
            // and output (data) buffers are available for writing
            while (!(inputDataAvailable() && outputDataAvailable()));

            // Lock buffers
            // To support data copy overlapping with the CNN execution,
            // the I/O buffers are locked the whole duration of the Subnet node execution
            // Each buffer lock is created as a scope-lock. Using a mutex (defined within a
            // buffer) such a lock locks the buffer on the lock creation and releases the
            // buffer on lock destruction. Note, that the locks are created in a loop
            // that traverses all input or all output buffers. To avoid the locks destruction
            // upon the loop end (we want the locks to be destroyed after the buffers
            // released only in the end of the run iteration), the locks are put in vectors.

            // std::cout<<"Before buffers locked..."<<std::endl;

            // lock input buffers for reading
            std::vector<readingLock> readingLocks;
            for (auto bufPtr: inputBufferPtrs){
                readingLock lock = bufPtr->lockForReading();
                readingLocks.push_back(std::move(lock));
            }

            // lock output buffers for writing
            std::vector<updatesLock> updateLocks;
            for (auto bufPtr: outputBufferPtrs){
                updatesLock lock = bufPtr->lockForWriting();
                updateLocks.push_back(std::move(lock));
            }
	
	     ///////////////////
            // perform reading
            for (auto bufPtr: inputBufferPtrs){
                bufPtr->read();
                bufPtr->syncAfterReading();
            }

            ///////////////////
      	    // execute dnn Inference
      	    this->dnn_ptr->do_run(); 
      	    // std::cout<<"CPU ENGINE "<<this->name<<" executed! "<<std::endl;
 	
      	    //////////////////
      	    // perform writing
            for (auto bufPtr: outputBufferPtrs){
                bufPtr->write();
                bufPtr->syncAfterWriting();
            }

            // this->dnn_ptr->doWrite(this->output, this->cuda_stream_ptr);
            // Buffer locks will be automatically released here after
            // readingLocks and  updateLocks vectors reach the scope end and are cleared
        }
  }

  catch(std::runtime_error &err){
    std::cerr << std::endl<< "CPU ENGINE ERROR " << err.what() << " " << (errno ? strerror(errno) : "") << std::endl;
    return;
  }
}
