#include "common.h"
#include "cuda_runtime_api.h"
#include <map>
#include <vector>
#include <thread>
#include "types.h"
#include "fifo.h"
#include <string>
#include "gpu_engine.h"

using namespace std;

/** GPU time profiler */
struct Profiler : public IProfiler
{
    typedef std::pair<std::string, float> Record;
    std::vector<Record> mProfile;

    virtual void reportLayerTime(const char* layerName, float ms)
    {
        auto record = std::find_if(mProfile.begin(), mProfile.end(), [&](const Record& r){ return r.first == layerName; });
        if (record == mProfile.end())
            mProfile.push_back(std::make_pair(layerName, ms));
        else
            record->second += ms;
    }


    void printLayerTimes()
    {
        float totalTime = 0;
        for (size_t i = 0; i < mProfile.size(); i++)
        {
            printf("%-40.40s %4.3fms\n", mProfile[i].first.c_str(), mProfile[i].second / frames);
            totalTime += mProfile[i].second;
        }
        printf("Time over all layers: %4.3f\n", totalTime / frames);
    }

    
    void generateEvalJSON(std::string filepath){
       ofstream eval_file;
       eval_file.open(filepath);
       int comma_ctr = 0;
       int max_comma_ctr = mProfile.size() - 1;
       
       // Print layer names
       eval_file << "{\n\"layers\": [";
       for( auto record : mProfile){
            std::string layer_name = record.first.c_str();
            // replace merge ( + ) symbol, incorrect for json with symbol "_" 
            int plus_id = layer_name.find(" + ");
            if(plus_id!=string::npos)
                layer_name.replace(plus_id, 3, "_");
            eval_file <<"\"" << layer_name << "\"";
            if(comma_ctr<max_comma_ctr)
               eval_file << ", ";
            comma_ctr++;
        }
       eval_file << "], ";
       
       // Print execution times
       comma_ctr = 0;
       eval_file << "\n\"GPU\": [";
       for( auto record : mProfile){
            float time = (record.second) / frames;
        
            eval_file << time;
            if(comma_ctr<max_comma_ctr)
               eval_file << ", ";
            comma_ctr++;
        }
        eval_file << "]\n}";
       eval_file.close();

    }    

    void printLayerNames()
    {   
        
        for( auto record : mProfile){
            std::string layer_name = record.first.c_str();
            // replace merge ( + ) symbol, incorrect for json with symbol "_" 
            int plus_id = layer_name.find(" + ");
            if(plus_id!=string::npos)
                layer_name.replace(plus_id, 3, "_");
            
            std::cout<<layer_name<<std::endl;
        }
    }

    float getLayerTime(const char* layerName)
    {   
        auto record = std::find_if(mProfile.begin(), mProfile.end(), [&](const Record& r){ return r.first == layerName; });
        return (record->second) / frames;
    }

} gProfiler;

/** CONSTRUCTOR**/
gpu_engine::gpu_engine(gpu_partition* dnn_ptr, float* input, float *output, cudaStream_t* stream_ptr, std::string name){
   this->dnn_ptr = dnn_ptr;
   this->input = input;
   this->output = output;
   this->cuda_stream_ptr = stream_ptr;
   this->name = name;

   bool setup_err = false;

   if(dnn_ptr==nullptr){
        std::cerr << std::endl<< "GPU ENGINE "<<this->name<<" SETUP ERROR: DNN PARTITION PTR IS NULL" << std::endl;
        setup_err=true;
   }

   if(input==nullptr){
        std::cerr << std::endl<< "GPU ENGINE "<<this->name<<" SETUP ERROR: DNN INPUT BUFFER PTR IS NULL" << std::endl;
        setup_err=true;
   }

   if(output==nullptr){
        std::cerr << std::endl<< "GPU ENGINE "<<this->name<<" SETUP ERROR: OUTPUT BUFFER PTR IS NULL" << std::endl;
        setup_err=true;
   }
   
   if(!setup_err)  
        std::cout<<"GPU ENGINE "<<this->name<<" CREATED!"<<std::endl;
}

/** DESTRUCTOR **/
gpu_engine::~gpu_engine(){

}


/** INFERENCE HERE **/
void gpu_engine::main(void *vpar) {
  try{   
        thread_info* par = (struct thread_info *) vpar;
  	setaffinity(par->core_id);

        fifo_buf* in_buf_ptr  = par->get_fifo_buf_by_dst(this->name);
        fifo_buf* out_buf_ptr = par->get_fifo_buf_by_src(this->name);

        /**set profiler, if required*/
        if (this->dnn_ptr->detailed_profile)
        	this->dnn_ptr->context->setProfiler(&gProfiler);

        //in case Multi I/Os needed
        //std::vector<fifo_buf*> get_in_fifos(this->name);
        //std::vector<fifo_buf*> get_out_fifos(this->name);

        //max tokens port IP0
        int IP0_tokens = 0;
        int OP0_tokens = 0;

        /**if(in_buf_ptr == nullptr) std::cout<<this->name<<" has null input buffer."<<std::endl;
        else IP0_tokens = in_buf_ptr->in_rate;

        if(out_buf_ptr == nullptr) std::cout<<this->name<<" has null output buffer."<<std::endl;
        else OP0_tokens = out_buf_ptr->out_rate;*/

  	//auto startTime = std::chrono::high_resolution_clock::now();
        
        for (int i=0; i<frames;i++){
          //read
           if ( IP0_tokens > 0 )
               readSWF_CPU(in_buf_ptr->fifo, &input[0], IP0_tokens, in_buf_ptr->fifo_size);

           this->dnn_ptr->doRead(this->input, this->cuda_stream_ptr);
          // std::cout<<"GPU ENGINE "<<this->name<<" reads "<<this->dnn_ptr->batchSize * this->dnn_ptr->INPUT_H * this->dnn_ptr->INPUT_W * this->dnn_ptr->INPUT_C<<" from input buffer "<<std::endl;
      	   this->dnn_ptr->doInference(this->cuda_stream_ptr);
           this->dnn_ptr->doWrite(this->output, this->cuda_stream_ptr);
          // std::cout<<"GPU ENGINE "<<this->name<<" writes "<<this->dnn_ptr->OUTPUT_SIZE<<" to output buffer "<<std::endl;
           this->dnn_ptr->doSync(this->cuda_stream_ptr);
          //std::cout<<"GPU ENGINE "<<this->name<<" executed! "<<std::endl;
          //write
           if ( OP0_tokens > 0 )
	      writeSWF_CPU(out_buf_ptr->fifo, &output[0], OP0_tokens, out_buf_ptr->fifo_size); 

        }

	/** print layers time profile information to console and to file*/ 
        if (this->dnn_ptr->detailed_profile) {
        	gProfiler.printLayerTimes();
        	gProfiler.generateEvalJSON ((this->name + ".json"));
        }
  	//auto endTime = std::chrono::high_resolution_clock::now();
  	//float totalTime = std::chrono::duration<float, std::milli>(endTime - startTime).count();
  	//std::cout<<this->name <<" (GPU + cpu core "<<par->core_id<<"): average over "<<frames<< " images = ~ "<<(totalTime/float(frames))<<" ms/img "<<std::endl;
 }

  catch(std::runtime_error &err){
   std::cerr << std::endl<< "GPU ENGINE ERROR " << err.what() << " " << (errno ? strerror(errno) : "") << std::endl;
   return;
 }
}
