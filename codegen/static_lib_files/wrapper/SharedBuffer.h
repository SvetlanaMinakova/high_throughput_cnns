//
// Created by svetlana on 11/04/2022.
//

#ifndef SH_BUF_JETSON_SHAREDBUFFER_H
#define SH_BUF_JETSON_SHAREDBUFFER_H

#include <string>
#include <memory>
#include <vector>
#include <shared_mutex>

class SharedBuffer {
/**A buffer that can by used by multiple processes
 * The buffer uses lock mechanism as proposed in
 * // https://riptutorial.com/cplusplus/example/30186/object-locking-for-efficient-access-
 * */
public:
    // attributes
    // buffer size (in tokens), i.e., maximum amount of data
    // elements that can be stored in buffers
    int size;
    std::string name;

    // constructor and destructor
    //constructor and destructor
    SharedBuffer();
    virtual void init(std::string name, int size);

    // mutex types aliases, given for code readability and maintainability
    using mutexType = std::shared_timed_mutex;
    using readingLock = std::shared_lock<mutexType>;
    using updatesLock = std::unique_lock<mutexType>;

    // synchronization (mutexes)
    // This returns a scoped lock that can be shared by multiple
    // readers at the same time while excluding any writers
    [[nodiscard]]
    virtual readingLock lockForReading() const { return readingLock(mtx); }

    // This returns a scoped lock that is excluding to one
    // writer preventing any readers
    [[nodiscard]]
    virtual updatesLock lockForWriting() { return updatesLock(mtx); }

    // read and write primitives
    virtual void read() = 0;
    virtual void write() = 0;

    virtual bool readyForReading() = 0;
    virtual bool readyForWriting() = 0;

    virtual void syncAfterReading()=0;
    virtual void syncAfterWriting()=0;

    virtual void copyToBuffer(float*data)=0;
    virtual void copyFromBuffer(float*data)=0;

    // checks on r/w availability

    /**
    // read and write simulation functions
    // these functions do not work with actual memory,
    // but change the buffer tokens counter as if data was read/written
    // from/to the buffer
    virtual void ReadSim(int data_tokens, int start_token=0);
    virtual void WriteSim(int data_tokens);
    // abstract data print function
    virtual void PrintData()=0;

    // general checks and flags
    bool IsEmpty();
    bool IsFull();
    int StoredTokens();
    int FreeTokens();
    bool IsVisited();
    void MarkAsVisited();
    void MarkAsUnVisited();
     */

protected:
    // number of currently stored data tokens
    int tokens = 0;
    bool visited = false;
    mutable mutexType mtx; // mutable allows const objects to be locked
};

#endif //SH_BUF_JETSON_SHAREDBUFFER_H
