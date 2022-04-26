//
// Created by svetlana on 11/04/2022.
//

#ifndef SH_BUF_JETSON_TYPES_H
#define SH_BUF_JETSON_TYPES_H

#include <pthread.h>
#include <vector>
#include <cstddef>
#include <string>
#include <memory>
#include <map>

struct ThreadInfo{
    char *message;
    pthread_t thread_id; // ID returned by pthread_create()
    int core_id; // Core ID we want this pthread to set its affinity to
    //references to shared FIFO buffers
    // std::vector<fifo_buf*> fifo_refs;
};

void setaffinity(int core);


#endif //SH_BUF_JETSON_TYPES_H
