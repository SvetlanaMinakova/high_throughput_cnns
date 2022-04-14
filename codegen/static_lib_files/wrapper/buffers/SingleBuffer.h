//
// Created by svetlana on 11/04/2022.
//

#ifndef SH_BUF_JETSON_SDFBINARYSTATEBUFFER_H
#define SH_BUF_JETSON_SDFBINARYSTATEBUFFER_H

#include <string>
#include "SharedBuffer.h"

class SingleBuffer: public SharedBuffer{
public:
    bool full = false;

    // constructor and destructor
    explicit SingleBuffer(): SharedBuffer(){
        this->full = false;
        this->data = nullptr;
    }
    void init(std::string name, int size) override;
    ~SingleBuffer();

    void read() override;
    void write() override;

    bool readyForReading() override;
    bool readyForWriting() override;

    // no special sync after r/w is required
    void syncAfterReading() override{};
    void syncAfterWriting() override{};

    void copyToBuffer(float* data) override;
    void copyFromBuffer(float* data) override;

    float *data;
};

#endif //SH_BUF_JETSON_SDFBINARYSTATEBUFFER_H
