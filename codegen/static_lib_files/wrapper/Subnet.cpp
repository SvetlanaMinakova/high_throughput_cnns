//
// Created by svetlana on 11/04/2022.
//

#include "Subnet.h"
#include "vector"
#include <utility>
#include <iostream>
#include "types.h"
#include <chrono>
#include <thread>

Subnet::Subnet(std::string name, int runs, int execDelay, int rwDelay) {
    this->name = std::move(name);
    this->runs = runs;
    this->execDelayMS = execDelay;
    this->rwDelayMS = rwDelay;
}

// buffers processing
void Subnet::addInputBufferPtr(SharedBuffer *ptr) {
    inputBufferPtrs.push_back(ptr);
}

void Subnet::addOutputBufferPtr(SharedBuffer *ptr) {
    outputBufferPtrs.push_back(ptr);
}

void Subnet::printInputBufferNames() {
    for (auto buf:inputBufferPtrs){
        std::cout<<buf->name<<std::endl;
    }
}

void Subnet::printOutputBufferNames() {
    for (auto buf:outputBufferPtrs){
        std::cout<<buf->name<<std::endl;
    }
}

// read, write, execute primitives

void Subnet::read(){

    //wait until all input data is available
    while (!inputDataAvailable());

    for (auto bufPtr:inputBufferPtrs){
        // lock for reading
        auto bufLock = bufPtr->lockForReading();
        bufPtr->read();
    }
}


void Subnet::write(){
    //wait until all output data is available
    while (!outputDataAvailable());

    for (auto bufPtr:outputBufferPtrs){
        // lock for writing
        auto bufLock = bufPtr->lockForWriting();
        bufPtr->write();
    }
}

void Subnet::delay(int timeMS) {
    if (timeMS>0)
        std::this_thread::sleep_for(std::chrono::milliseconds(timeMS));
}

void Subnet::exec() {
    std::cout<<"Execute "<< this->name<<std::endl;
}

bool Subnet::inputDataAvailable() {
    for (auto bufPtr: inputBufferPtrs){
        if(!bufPtr->readyForReading())
            return false;
    }
    return true;
}

bool Subnet::outputDataAvailable() {
    for(auto bufPtr:outputBufferPtrs){
        if (!bufPtr->readyForWriting())
            return false;
    }
    return true;
}

// locks
Subnet::readingLock Subnet::lockBufferForReading(SingleBuffer *bufPtr) {
    return bufPtr->lockForReading();
}

// in double buffer reading is always done from the bottom
Subnet::readingLock Subnet::lockBufferForReading(DoubleBuffer *bufPtr) {
    return bufPtr->bottom->lockForReading();
}

Subnet::updatesLock Subnet::lockBufferForWriting(SingleBuffer *bufPtr) {
    return bufPtr->lockForWriting();
}

// in double buffer writing is always done to the top
Subnet::updatesLock Subnet::lockBufferForWriting(DoubleBuffer *bufPtr) {
    return bufPtr->top->lockForWriting();
}


void Subnet::main(void *thread_par) {
    // Lock buffers - Generic function?
    // std::cout<<"Subnet main entered!"<<std::endl;
    for (int run=0; run<runs; run++){
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

        std::cout<<"Locks created!"<<std::endl;

        // perform reading
        for (auto bufPtr: inputBufferPtrs){
            bufPtr->read();
            bufPtr->syncAfterReading();
        }

        // delay execution
        delay(execDelayMS);

        // execute
        exec();

        // perform writing
        for (auto bufPtr: outputBufferPtrs){
            bufPtr->write();
            bufPtr->syncAfterWriting();
        }

        // Buffer locks will be automatically released here after
        // readingLocks and  updateLocks vectors reach the scope end and are cleared
    }
}
