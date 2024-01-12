#ifndef __MYTENSORRT_H__
#define __MYTENSORRT_H__
#include <NvInferRuntime.h>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cassert>
#include <memory>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <chrono>
#include "Logger.h"

typedef struct Data_s {
    void *dataptr;
    int size;
    virtual ~Data_s() {} // 添加虚析构函数
}Data;
// bool exists(const std::string& path);
class TensorRTModel : public Logger {
// private:
protected:
    size_t buffersize;
    void *host_buffer=nullptr;
    void *device_buffer=nullptr;
    char *host_buffer_now=nullptr;
    char *device_buffer_now=nullptr;
    cudaStream_t stream;
    nvinfer1::IRuntime *runtime;
    nvinfer1::ICudaEngine *engine;
    nvinfer1::IExecutionContext *context;
    void *image_data_device = nullptr;
    void *input_data_host = nullptr;
    void *input_data_device = nullptr;
    void *output_data_host = nullptr;
    void *output_data_device = nullptr;
    void *h_mean = nullptr;
    void *h_std = nullptr;
    void *d_mean = nullptr;
    void *d_std = nullptr;
    void *h_matrix = nullptr;
    void *d_matrix = nullptr;
    std::string onnx_file;
    std::string trt_file;
    std::string name;
    void* h_filtered_boxes = nullptr;
    void* d_filtered_boxes = nullptr;
    void* h_box_count = nullptr;
    void* d_box_count = nullptr;
    int input_batch;
    int input_numel;
    int output_batch;
    int output_numbox;
    int output_numprob;
    int num_classes;
    int output_numel;
    int index = 0;
    int optBatchSize = 1;
    int maxBatchSize = 1;
    int channelSize = 3;
    int inputW = 640;
    int inputH = 640;
    bool use_int8 = false;
    bool use_fp16 = false;
    bool set_memory = false;
    bool init_finished = false;
    // static bool build_finished = false;
    bool build_finished = false;
    int temp = 1024;

public:
    TensorRTModel();
    TensorRTModel(const std::string &name);
    TensorRTModel(const std::string &name, int buffer_size);
    virtual ~TensorRTModel();

    int malloc_host(void **ptr, size_t size);
    int malloc_device(void** ptr, size_t size);

    // int build();
    virtual int build() = 0;
    virtual int init() = 0;
    virtual void reset() = 0;
    virtual int setMemory(void *cpuptr, void *gpuptr, int buffersize) = 0;
    virtual int preprocess(cv::Mat &img) = 0;
    virtual int postprocess() = 0;
    virtual int forward() = 0;
    virtual int inference(Data *data) = 0;
};

#endif //__MYTENSORRT_H__