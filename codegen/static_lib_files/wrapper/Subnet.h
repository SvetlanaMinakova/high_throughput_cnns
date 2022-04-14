//
// Created by svetlana on 11/04/2022.
//

#ifndef SH_BUF_JETSON_SDFPROCESS_H
#define SH_BUF_JETSON_SDFPROCESS_H

#include <string>
#include "buffers/SharedBuffer.h"
#include "buffers/SingleBuffer.h"
#include "buffers/DoubleBuffer.h"
#include <vector>
#include <memory>
#include <shared_mutex>

class Subnet {
public:
    std::string name;
    int execDelayMS=0;
    int rwDelayMS=0;
    int runs;

    // mutex types aliases, given for code readability and maintainability
    using mutexType = std::shared_timed_mutex;
    using readingLock = std::shared_lock<mutexType>;
    using updatesLock = std::unique_lock<mutexType>;

    // constructor and destructor
    explicit Subnet(std::string name, int runs=1, int execDelayMS=0, int rwDelayMS=0);

    void delay(int timeMS);
    virtual void exec();
    virtual void read();
    virtual void write();

    virtual void addInputBufferPtr(SharedBuffer* ptr);
    virtual void addOutputBufferPtr(SharedBuffer* ptr);
    void printInputBufferNames();
    void printOutputBufferNames();

    virtual bool inputDataAvailable();
    virtual bool outputDataAvailable();

    // main function executed in a thread
    virtual void main(void *thread_par);

    // mutex locks
    static readingLock lockBufferForReading(SingleBuffer* bufPtr);
    static readingLock lockBufferForReading(DoubleBuffer* bufPtr);
    static updatesLock lockBufferForWriting (SingleBuffer* bufPtr);
    static updatesLock lockBufferForWriting (DoubleBuffer* bufPtr);

protected:
    std::vector<SharedBuffer*> inputBufferPtrs;
    std::vector<SharedBuffer*> outputBufferPtrs;
};


#endif //SH_BUF_JETSON_SDFPROCESS_H
