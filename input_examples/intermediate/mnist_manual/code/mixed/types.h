//File automatic generated by espamAI

#ifndef types_H
#define types_H

//#include "arm_compute/graph.h"
#include <pthread.h>
#include <vector>
#include <cstddef>
#include <string>
#include <memory>
#include <map>

#define frames 50

/** description of buffer of one FIFO channel*/
struct fifo_buf{
	std::string src; //fifo src name
	std::string dst; //fifo dst name
	void* fifo; //ptr to shared memory
	int fifo_size; // size of the buffer (in tokens)
        int in_rate; //input tokes per one input reading
        int out_rate; //output tokens per one output writing

	fifo_buf(void* fifo, int fifo_size, std::string src, std:: string dst, int in_rate, int out_rate){
		this->fifo = fifo;
		this->fifo_size = fifo_size;
		this->src = src;
		this->dst = dst;     
                this->in_rate = in_rate;
                this->out_rate = out_rate;           
	}
};


struct thread_info{

  char *message;
  pthread_t thread_id; // ID returned by pthread_create()
  int core_id; // Core ID we want this pthread to set its affinity to
  //references to fifos
  std::vector<fifo_buf*> fifo_refs;

  // get fifo by source
  fifo_buf* get_fifo_buf_by_src(std::string srcname){
	  for (auto & fifos_elem : fifo_refs) {
		  if(srcname.compare(fifos_elem->src) == 0)
			  return fifos_elem;
		  }
		  return nullptr;
  }

  // get output fifos by source node name
  std::vector<fifo_buf*> get_out_fifos(std::string srcname){
  	  std::vector<fifo_buf*> out_fifos; 
	  for (auto & fifos_elem : fifo_refs) {
		  if(srcname.compare(fifos_elem->dst) == 0)
			  out_fifos.push_back(fifos_elem);
		  }
	  return out_fifos;
  }

  // get fifo by name
  fifo_buf* get_fifo_buf_by_dst(std::string dstname){
	  for (auto & fifos_elem : fifo_refs) {
		  if(dstname.compare(fifos_elem->dst) == 0)
			  return fifos_elem;
		  }
		  return nullptr;
  }

  // get input fifos by destination node name
  std::vector<fifo_buf*> get_in_fifos(std::string dstname){
          std::vector<fifo_buf*> in_fifos; 
	  for (auto & fifos_elem : fifo_refs) {
                  if(dstname.compare(fifos_elem->dst) == 0)
			  in_fifos.push_back(fifos_elem);
		  }
	  return in_fifos;
  }

  void add_fifo_buf_ref(fifo_buf* fifo_buf_ref){
	  fifo_refs.push_back(fifo_buf_ref);
  }
};

/** inference arguments structure
struct inferParams{
        int runIter;
        int id;
        int coreId;
        
        inferParams(
        int RunIter,
        int Id,
        int CoreId){
           runIter = RunIter;
           id = Id;
           coreId = CoreId;
        }
};

/** inference arguments structure
struct inferParamsv2{
        Example* dnn_ptr;
        int runIter;
        int id;
        int coreId;
        
        inferParamsv2(
        Example* Dnn_ptr,
        int RunIter,
        int Id,
        int CoreId){
           dnn_ptr = Dnn_ptr;
           runIter = RunIter;
           id = Id;
           coreId = CoreId;
        }
};*/

#endif // types_H