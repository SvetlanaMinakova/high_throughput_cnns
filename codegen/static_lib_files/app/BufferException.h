//
// Created by svetlana on 11/04/2022.
//

#ifndef SH_BUF_JETSON_BUFFEREXCEPTION_H
#define SH_BUF_JETSON_BUFFEREXCEPTION_H

#include <iostream>
#include <exception>
#include <string>

struct BufferException: public std::exception {
protected:
    const char* message = "BufferException";
public:
    explicit BufferException(const char* message){
        this->message = message;
    }
    [[nodiscard]] const char * what () const noexcept override
    {
        return message;
    }
};

#endif //SH_BUF_JETSON_BUFFEREXCEPTION_H
