// capture.py's log(): local time as "%I:%M:%S %p", then the message. One
// write per line so threads never interleave inside a line.
#pragma once

#include <ctime>
#include <iostream>
#include <mutex>
#include <string_view>

namespace bunnycam {

inline void log(std::string_view message) {
    static std::mutex mu;
    char stamp[16];
    std::time_t t = std::time(nullptr);
    std::tm local{};
    localtime_r(&t, &local);
    std::strftime(stamp, sizeof stamp, "%I:%M:%S %p", &local);
    std::lock_guard lock(mu);
    std::cout << stamp << ' ' << message << std::endl;
}

}  // namespace bunnycam
