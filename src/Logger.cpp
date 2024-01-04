//
// Created by wen on 2023-12-30.
//

// Logger.cpp
#include "Logger.h"

//Logger::Logger() : out_stream(&std::cout), currentLevel(level){}
Logger::Logger(LogLevel level) : out_stream(&std::cout), currentLevel(level) {}
void Logger::SetLevel(LogLevel level) {
    currentLevel = level;
}

void Logger::SetOutput(std::ostream* os) {
    out_stream = os;
}
void Logger::Log(LogLevel level, const char* file, int line, const char* func, const char* format, ...) {
    if (level < currentLevel) return;
    std::lock_guard<std::mutex> lock(mu);
    // ... [时间和颜色处理不变]
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto localTime = *std::localtime(&time);

    *out_stream << "[" << std::put_time(&localTime, "%Y-%m-%d %H:%M:%S") << "] ";
#if 0
    switch (level) {
        case LogLevel::DEBUG:
            *out_stream << "[DEBUG] ";
            break;
        case LogLevel::INFO:
            *out_stream << "[INFO] ";
            break;
        case LogLevel::WARNING:
            *out_stream << "[WARNING] ";
            break;
        case LogLevel::ERROR:
            *out_stream << "[ERROR] ";
            break;
        case LogLevel::FATAL:
            *out_stream << "[FATAL] ";
            break;
    }
    *out_stream << "[" << file << ":" << line << ":" << func << "] ";
    char buffer[1024];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);

    *out_stream << buffer  << std::endl;
#else
    switch (level) {
        case LogLevel::DEBUG:
            *out_stream << BLUE << "[DEBUG] " << RESET;
            break;
        case LogLevel::INFO:
            *out_stream << GREEN << "[INFO] " << RESET;
            break;
        case LogLevel::WARNING:
            *out_stream << YELLOW << "[WARNING] " << RESET;
            break;
        case LogLevel::ERROR:
            *out_stream << RED << "[ERROR] " << RESET;
            break;
        case LogLevel::FATAL:
            *out_stream << PURPLE << "[FATAL] " << RESET;
            break;
    }
    *out_stream << "[" << file << ":" << line << ":" << func << "] ";

    char buffer[1024];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);

    *out_stream << buffer << RESET << std::endl;
#endif
}
void Logger::Log(const std::string& message, LogLevel level,
                 const char* file, int line, const char* func) {
    if (level < currentLevel) return;
    std::lock_guard<std::mutex> lock(mu);
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto localTime = *std::localtime(&time);

    *out_stream << "[" << std::put_time(&localTime, "%Y-%m-%d %H:%M:%S") << "] ";
#if 0
    switch (level) {
        case LogLevel::DEBUG:
            *out_stream << "[DEBUG] ";
            break;
        case LogLevel::INFO:
            *out_stream << "[INFO] ";
            break;
        case LogLevel::WARNING:
            *out_stream << "[WARNING] ";
            break;
        case LogLevel::ERROR:
            *out_stream << "[ERROR] ";
            break;
        case LogLevel::FATAL:
            *out_stream << "[FATAL] ";
            break;
    }
    *out_stream << "[" << file << ":" << line << ":" << func << "] " << message << std::endl;
#else
    switch (level) {
        case LogLevel::DEBUG:
            *out_stream << BLUE << "[DEBUG] " << RESET;
            break;
        case LogLevel::INFO:
            *out_stream << GREEN << "[INFO] " << RESET;
            break;
        case LogLevel::WARNING:
            *out_stream << YELLOW << "[WARNING] " << RESET;
            break;
        case LogLevel::ERROR:
            *out_stream << RED << "[ERROR] " << RESET;
            break;
        case LogLevel::FATAL:
            *out_stream << PURPLE << "[FATAL] " << RESET;
            break;
    }
    *out_stream << "[" << file << ":" << line << ":" << func << "] " << message << RESET << std::endl;
#endif
}