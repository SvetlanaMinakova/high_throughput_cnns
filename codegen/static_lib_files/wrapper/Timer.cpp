//
// Created by svetlana on 14/04/22.
//

#include "Timer.h"
#include <iostream>
#include <chrono>
#include <ctime>   // localtime
#include <sstream> // stringstream
#include <iomanip> // put_time


std::string Timer::currentTimeAndDateStr() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");
    return ss.str();
}

void Timer::printCurrentTimeAndDate() {
    std::string timeAndDate = currentTimeAndDateStr();
    std::cout<<timeAndDate<<std::endl;
}

double Timer::timeElapsed(std::chrono::time_point<std::chrono::system_clock> start,
                          std::chrono::time_point<std::chrono::system_clock> end) {
    std::chrono::duration<double> execTimeDiff = end - start;
    double execTimeS = execTimeDiff.count();
    return execTimeS;
}
