#ifndef __YOLO_H__
#define __YOLO_H__

#include "MyTensorrt.h"
#include "Logger.h"
// #include "yolov5_config.h"
#include "preprocess_gpu.cuh"
#include "postprocess_gpu.cuh"
#include <cstdlib>
#include <chrono>
#include "preprocess.h"
#include "postprocess.h"
// #include "yolov5_utils.h"
// #include "mempool.h"
// #include "mempool_gpu.h"

typedef struct ImageData:Data {
    int width;
    int height;
    int channels;
    int batch;
    int numel;
}ImageData_t;

class Yolov5:public MyTensorRT, public Logger {
public:
    Yolov5();
    Yolov5(std::string& name);
    Yolov5(std::string& name, int buffer_size);
    ~Yolov5();
    int init() override;
    int setMemory(void *cpuptr, void *gpuptr, int buffersize) override;
    int preprocess(cv::Mat &img) override;
    int preprocess(ImageData_t *imgdata);
    int postprocess() override;
    void drawimg(cv::Mat &img, const std::string& savepath);
    int forward() override;
    int inference(Data* data) override;
    int inference_image(std::string &img_file);
    void make_imagedata(const cv::Mat& image, ImageData_t* imagedata);
    // int malloc_host(void **ptr, size_t size);
    // int malloc_device(void** ptr, size_t size);
    // int free_host(void *ptr);
    // int free_device(void * ptr);
// private:
    // MemPool mempool;
    // MemPoolGpu mempool_gpu;
// private:
//     std::unique_ptr<MemoryPool> mempool;
//     std::unique_ptr<MemoryPoolGpu> mempool_gpu;
};

#endif // __YOLO_H_