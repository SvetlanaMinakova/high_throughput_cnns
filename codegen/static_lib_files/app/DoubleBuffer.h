//
// Created by svetlana on 11/04/2022.
//

#ifndef SH_BUF_JETSON_SDFBINARYSTATEDOUBLEBUFFER_H
#define SH_BUF_JETSON_SDFBINARYSTATEDOUBLEBUFFER_H

#include "SharedBuffer.h"
#include "SingleBuffer.h"

class DoubleBuffer: public SharedBuffer{
public:
    SingleBuffer* top;
    SingleBuffer* bottom;

    // mutex types aliases, given for code readability and maintainability
    using mutexType = std::shared_timed_mutex;
    using readingLock = std::shared_lock<mutexType>;
    using updatesLock = std::unique_lock<mutexType>;

    // synchronization (mutexes)
    // This returns a scoped lock that can be shared by multiple
    // readers at the same time while excluding any writers
    [[nodiscard]]
    readingLock lockForReading() const override { return bottom->lockForReading(); }

    // This returns a scoped lock that is excluding to one
    // writer preventing any readers
    [[nodiscard]]
    updatesLock lockForWriting() override { return top->lockForWriting(); }

    // This returns a scoped lock that is excluding to one
    // writer preventing any readers
    [[nodiscard]]
    updatesLock lockForSwapping() { return updatesLock(mtx); }

    // constructor and destructor
    explicit DoubleBuffer(): SharedBuffer(){
        top = &buf1;
        bottom = &buf2;
    }
    void init(std::string name, int size) override;

    // reading and writing
    void read() override;
    void write() override;

    virtual void swap();

    bool readyForReading() override;
    bool readyForWriting() override;
    bool readyForSwapping();

    void syncAfterReading() override;
    void syncAfterWriting() override;

    void copyToBuffer(float* data) override;
    void copyFromBuffer(float* data) override;

protected:
    SingleBuffer buf1;
    SingleBuffer buf2;
};


#endif //SH_BUF_JETSON_SDFBINARYSTATEDOUBLEBUFFER_H
