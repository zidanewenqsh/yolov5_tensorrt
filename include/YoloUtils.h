#ifndef __YOLOUTILS_H__
#define __YOLOUTILS_H__
#include <iostream>
#include <vector>
#include <fstream>
#include <memory>
#include <cassert>
#include <NvInfer.h>
#include <unistd.h>

#include <chrono>
#include <iomanip>
#include <sstream>
#include "MySQLConnPool.h"
// #include <filesystem> // c++17
void insertImageInfo(const std::shared_ptr<sql::Connection>& conn, const std::string& file_name, const std::string& upload_time, const std::string& source);
void insertDetectionResult(const std::shared_ptr<sql::Connection>& conn, int image_id, const std::string& class_name, double confidence, int bbox_xmin, int bbox_ymin, int bbox_xmax, int bbox_ymax, const std::string& detection_time);
std::string getCurrentDateTime();
bool exists(const std::string& path);
void saveWeight(const std::string &filename, const float *data, const int size);
std::vector<float> loadWeight(const std::string &filename);
std::vector<std::string> readLinesFromFile(const std::string& filename);
void saveData(const std::string &filename, const char *data, const int size);
std::vector<char> loadData(const std::string &filename);
void printDims(const nvinfer1::Dims& dims);
void calculate_matrix(float* matrix, int width, int height, int w, int h);
std::string extractFilename(const std::string& filePath);
std::string extractFilenameWithoutExtension(const std::string& filePath);
#endif // __YOLOUTILS_H__