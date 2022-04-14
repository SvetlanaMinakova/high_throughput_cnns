//
// Created by svetlana on 11/04/2022.
//

#include "SingleBuffer.h"
#include "BufferException.h"

void SingleBuffer::read() {
    if (!readyForReading())
        throw BufferException("Cannot read from unready BinaryState buffer!");
    this->full = false;
}

void SingleBuffer::write() {
    if (!readyForWriting())
        throw BufferException("Cannot write to unready BinaryState buffer!");
    this->full = true;
}

bool SingleBuffer::readyForReading() {
    // buffer is ready for reading when it is full
    return this->full;
}

bool SingleBuffer::readyForWriting() {
    // buffer is ready for reading when it is empty (not full)
    return !(this->full);
}

void SingleBuffer::init(std::string name, int size) {
    SharedBuffer::init(name, size);
    this-> data = new float [size];
}

SingleBuffer::~SingleBuffer() {
    delete [] this->data;
}

void SingleBuffer::copyToBuffer(float *inputData) {
    //read();
    for (int i=0; i<size; i++)
        this->data[i] = inputData[i];

}

void SingleBuffer::copyFromBuffer(float *outputData) {
    //write();
    for (int i=0; i<size; i++)
        outputData[i] = this->data[i];
}
