//
// Created by svetlana on 11/04/2022.
//

#include "types.h"

// CPU affinity mask of the thread to the CPU core
void setaffinity(int core){

    pthread_t pid = pthread_self();
    int core_id = core;

//   cpu_set_t: This data set is a bitset where each bit represents a CPU.
    cpu_set_t cpuset;
//  CPU_ZERO: This macro initializes the CPU set set to be the empty set.
    CPU_ZERO(&cpuset);
//   CPU_SET: This macro adds cpu to the CPU set set.
    CPU_SET(core_id, &cpuset);

//   pthread_setaffinity_np: The pthread_setaffinity_np() function sets the CPU affinity mask of the thread thread to the CPU set pointed to by
//cpuset. If the call is successful, and the thread is not currently running on one of the CPUs in cpuset, then it is migrated to one of those CPUs.
    const int set_result = pthread_setaffinity_np(pid, sizeof(cpu_set_t), &cpuset);
    if (set_result != 0) {
        printf("pthread setaffinity failed!\n");
    }


    //  Check what is the actual affinity mask that was assigned to the thread.
    //  pthread_getaffinity_np: The pthread_getaffinity_np() function returns the CPU affinity mask of the thread thread in the buffer pointed to by cpuset.
    const int get_affinity = pthread_getaffinity_np(pid, sizeof(cpu_set_t), &cpuset);
    if (get_affinity != 0) {
        printf("pthread getaffinity failed!\n");
    }

    char *buffer;
    // CPU_ISSET: This macro returns a nonzero value (true) if cpu is a member of the CPU set set, and zero (false) otherwise.
    if (CPU_ISSET(core_id, &cpuset)) {
//    printf("Successfully set thread %u to cpu core %d\n", pid, core_id);
    } else {
//    printf("failed!\n");
    }
}