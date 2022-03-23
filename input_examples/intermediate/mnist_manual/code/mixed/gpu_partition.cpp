#include "NvInfer.h"
#include "common.h"
#include "cuda_runtime_api.h"
#include <map>
#include <vector>
#include <thread>
#include "gpu_partition.h"

using namespace std;
using namespace nvinfer1;

static Logger gLogger;

/** INIT DNN partition - assumes all DNN sizes and weights are set*/
void gpu_partition::init_partition() {
  
    // create a model using the API directly and serialize it to a stream
    IHostMemory* modelStream{nullptr};
    this->APIToModel(this->weightMap, this->batchSize, &(modelStream));
    assert(modelStream!= nullptr);

    // runtime 
    this->runtime = createInferRuntime(gLogger);
    assert(this->runtime != nullptr);

    // engine and context
    this->engine = this->runtime->deserializeCudaEngine(modelStream->data(), modelStream->size(), nullptr); //create engine
    assert(this->engine != nullptr);
    modelStream->destroy(); //destory model stream  
    this->context = this->engine->createExecutionContext(); //create execution context
    //context->setProfiler(&gProfiler);
    assert(this->context != nullptr); 

    std::cout<<"Engine created!"<<std::endl;
   
    // Create GPU buffers on device

    //void* buffers[this->IObindings];
    //assert(engine->getNbBindings() == 2);
    
    this->inputIndex = this->engine->getBindingIndex(this->INPUT_BLOB_NAME);
    this->outputIndex = this->engine->getBindingIndex(this->OUTPUT_BLOB_NAME);

    //remove after testing
    if(inputIndex == -1) std::cout<<"ERR: partition input buffer "<<this->INPUT_BLOB_NAME<<" not found!"<<std::endl;
    if(outputIndex == -1) std::cout<<"ERR: partition output buffer "<<this->OUTPUT_BLOB_NAME <<" not found!"<<std::endl;

    // Create GPU buffers on device
    CHECK(cudaMalloc(&(this->buffers[inputIndex]), this->batchSize * this->INPUT_H * this->INPUT_W * this->INPUT_C * sizeof(float)));

    int additionalInputIndex; 
    for (int i=0; i < this->ADDITIONAL_INPUT_BLOB_NAMES.size(); i++){
        //std::cout<<"create additional input blob"<<ADDITIONAL_INPUT_BLOB_NAMES.at(i)<<std::endl;
        additionalInputIndex = this->engine->getBindingIndex(ADDITIONAL_INPUT_BLOB_NAMES.at(i));
        CHECK(cudaMalloc(&(this->buffers[additionalInputIndex]), this->batchSize * this->ADDITIONAL_INPUT_H.at(i)* this->ADDITIONAL_INPUT_W.at(i) * this->ADDITIONAL_INPUT_C.at(i) * sizeof(float)));
    }

    CHECK(cudaMalloc(&(this->buffers[outputIndex]), this->batchSize * OUTPUT_SIZE * sizeof(float)));
}

//generate dummy weights
// names : list of weights for partition
// sizes : sizes of weights for partition

//void gpu_partition::generate_dummy_weights(std::vector<std::string> names, std::vector<uint32_t> sizes) {

void gpu_partition::generate_dummy(std::vector<std::string> names, std::vector<uint32_t> sizes){ 
for (int count=0; count < names.size(); count++)
    {
        Weights wt{nvinfer1::DataType::kFLOAT, nullptr, 0};
        //uint32_t type, size;
        
        // Read name and type of blob
        std::string name = names.at(count);
        uint32_t size = sizes.at(count);

        //std::cout<<name<<" : "<<size<<std::endl;
        //input >> name >> std::dec >> type >> size;
        wt.type = nvinfer1::DataType::kFLOAT; //static_cast<nvinfer1::DataType>(type);

        if(size == 0) {
           wt.values = nullptr;
        }

        else {

        	// Load blob
        	if (wt.type == nvinfer1::DataType::kFLOAT)
        	{
            	uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
            	for (uint32_t x = 0, y = size; x < y; ++x)
            	{
                 	val[x] = (float)x * 0.01;
            	}
            	wt.values = val;
        	}

            else if (wt.type == nvinfer1::DataType::kHALF)
             {
            	uint16_t* val = reinterpret_cast<uint16_t*>(malloc(sizeof(val) * size));
            	for (uint32_t x = 0, y = size; x < y; ++x)
            		{
                	val[x] = (float)x * 0.01;
            		}
            	wt.values = val;
        	}
        }

        wt.count = size;
        this->weightMap[name] = wt;
    }
}


void gpu_partition::destroy_partition() {
  
    // Release weights
    for (auto& mem : this->weightMap)
    {
        free((void*) (mem.second.values));
    }
    //std::cout<<"Weights released!"<<std::endl;
    
    // Release buffers
    
    CHECK(cudaFree(this->buffers[this->inputIndex]));
    CHECK(cudaFree(this->buffers[this->outputIndex]));

    int additionalInputIndex; 
    for (int i=0; i < this->ADDITIONAL_INPUT_BLOB_NAMES.size(); i++){
        //std::cout<<"delete additional input blob"<<ADDITIONAL_INPUT_BLOB_NAMES.at(i)<<std::endl;
        additionalInputIndex = this->engine->getBindingIndex(ADDITIONAL_INPUT_BLOB_NAMES.at(i));
        CHECK(cudaFree(this->buffers[additionalInputIndex]));
    }
    //std::cout<<"Buffers released!"<<std::endl;

    // Destroy engine, context and runtime
    this->context->destroy();
    this->engine->destroy();
    this->runtime->destroy();

    //std::cout<<"GPU engine released!"<<std::endl;
}

gpu_partition::~gpu_partition() {
	destroy_partition();
};

/** CREATE ENGINE */
void gpu_partition::APIToModel(std::map<std::string, Weights>& weightMap, unsigned int maxBatchSize, IHostMemory** modelStream)
{
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(weightMap, maxBatchSize, builder, nvinfer1::DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}

/** DO READ **/
void gpu_partition::doRead(float* input, cudaStream_t* stream_ptr)
{    
     CHECK(cudaMemcpyAsync(this->buffers[this->inputIndex], input , batchSize * this->INPUT_H * this->INPUT_W * this->INPUT_C * sizeof(float), cudaMemcpyHostToDevice, *stream_ptr));
}

/** DO WRITE **/
void gpu_partition::doWrite(float* output, cudaStream_t* stream_ptr)
{    
     CHECK(cudaMemcpyAsync(output, this->buffers[this->outputIndex],  batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, *stream_ptr));
}

/** DO SYNC **/
void gpu_partition::doSync(cudaStream_t* stream_ptr)
{    
     cudaStreamSynchronize(*stream_ptr);
}

/** DO INFERENCE **/
void gpu_partition::doInference(cudaStream_t* stream_ptr)
{    
     this->context->enqueue(1, this->buffers, *stream_ptr, nullptr);
}
