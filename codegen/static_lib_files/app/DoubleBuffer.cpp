//
// Created by svetlana on 11/04/2022.
//

#include "DoubleBuffer.h"
#include <iostream>

void DoubleBuffer::init(std::string name, int size) {
    SharedBuffer::init(name, size);
    buf1.init("buf1", size);
    buf2.init("buf2", size);
}

bool DoubleBuffer::readyForReading() {
    return bottom->readyForReading();
}

void DoubleBuffer::read() {
    // std::cout<<"read from double buffer "<<name<<std::endl;
    bottom->read();
}

bool DoubleBuffer::readyForWriting() {
    return top->readyForWriting();
}

void DoubleBuffer::write() {
    // std::cout<<"write to double buffer "<<name<<std::endl;
    top->write();
}

bool DoubleBuffer::readyForSwapping() {
    // buffer is ready for swapping when top (data reader) is full and bottom (data writer) is empty
    if (top->full && !(bottom->full))
        return true;

    return false;
}

void DoubleBuffer::swap() {
    auto bufLock = this->lockForSwapping();
    SingleBuffer* tmp;
    tmp = bottom;
    bottom = top;
    top = tmp;
}

void DoubleBuffer::syncAfterReading() {
    if (readyForSwapping())
        swap();
}

void DoubleBuffer::syncAfterWriting() {
    if (readyForSwapping())
        swap();
}

void DoubleBuffer::copyToBuffer(float *data) {
    bottom->copyToBuffer(data);
}

void DoubleBuffer::copyFromBuffer(float *data) {
    top->copyFromBuffer(data);
}
