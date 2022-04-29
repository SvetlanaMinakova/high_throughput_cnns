#include "common.h"
#include "cuda_runtime_api.h"
#include <map>
#include <vector>
#include <thread>
#include "types.h"
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


void printLayerTimes(){
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
             // replace merge ( + ) symbol json with symbol "_" to allow for saving/parsing
            // resulting json
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

void printLayerNames(){
    for( auto record : mProfile){
        std::string layer_name = record.first.c_str();
        // replace merge ( + ) symbol json with symbol "_" to allow for saving/parsing
        // resulting json
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


/** DESTRUCTOR **/
gpu_engine::~gpu_engine(){

}


/** INFERENCE HERE **/
void gpu_engine::main(void *thread_par) {
    try{
        // assign CPU core to current thread
        auto* par = (struct ThreadInfo*) thread_par;
        setaffinity(par->core_id);

        // set TensorRT execution time profiler, if required
        if (this->dnn_ptr->detailed_profile)
        	this->dnn_ptr->context->setProfiler(&gProfiler);
        
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
	   
            //this->dnn_ptr->doRead(this->input, this->cuda_stream_ptr);
            // std::cout<<"GPU ENGINE "<<this->name<<" reads "<<this->dnn_ptr->batchSize * this->dnn_ptr->INPUT_H * this->dnn_ptr->INPUT_W * this->dnn_ptr->INPUT_C<<" from input buffer "<<std::endl;
        
            ///////////////////
      	    // execute dnn Inference
      	    this->dnn_ptr->doInference(this->cuda_stream_ptr);
       
           // std::cout<<"GPU ENGINE "<<this->name<<" writes "<<this->dnn_ptr->OUTPUT_SIZE<<" to output buffer "<<std::endl;
           this->dnn_ptr->doSync(this->cuda_stream_ptr);
           //std::cout<<"GPU ENGINE "<<this->name<<" executed! "<<std::endl;
           
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

        /** print layers time profile information to console and to file*/ 
        if (this->dnn_ptr->detailed_profile) {
            gProfiler.printLayerTimes();
            gProfiler.generateEvalJSON ((this->name + ".json"));
        }
    }

    catch(std::runtime_error &err){
        std::cerr << std::endl<< "GPU ENGINE ERROR " << err.what() << " " << (errno ? strerror(errno) : "") << std::endl;
        return;
    }
}
