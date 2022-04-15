//
// Created by svetlana on 11/04/2022.
//

#include "SharedBuffer.h"

// required for DoubleSharedBuffer
SharedBuffer::SharedBuffer() {
    this->name = "none";
    this->size = 0;
}

void SharedBuffer::init(std::string name, int size) {
    this->name=std::move(name);
    this->size=size;
}
