//
// Created by svetlana on 14/04/22.
//

#ifndef CNNPIPELINEINFERENCEWRAPPER_TIMER_H
#define CNNPIPELINEINFERENCEWRAPPER_TIMER_H

#include <string>
#include <chrono>

class Timer {
public:
    static std::string currentTimeAndDateStr();
    static void printCurrentTimeAndDate();
    static double timeElapsed(std::chrono::time_point<std::chrono::system_clock> start, std::chrono::time_point<std::chrono::system_clock> end);

};


#endif //CNNPIPELINEINFERENCEWRAPPER_TIMER_H
