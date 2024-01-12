#ifndef __INFERENCE_MANAGER_H__
#define __INFERENCE_MANAGER_H__
#include <cstddef>
#include <vector>
#include <future>
#include "ThreadPool.h" // 假设您的线程池定义在这个文件中
#include "YoloPool.h"   // 假设YoloPool的定义在这个文件中
#include "MemoryPool.h" // 假设MemoryPool的定义在这个文件中
#include "MemoryPoolGpu.h" // 假设MemoryPool的定义在这个文件中
#include <opencv2/opencv.hpp>
#include "Logger.h"
#include "Yolov5.h"
#include "YoloFactory.h"
#include "MySQLConnPool.h"

// #include "Image.h"      // 假设Image的定义在这个文件中
// #include "Result.h"     // 假设Result的定义在这个文件中
// #define Result int
typedef struct Image_s {
    cv::Mat img;
    std::string imgname;
    std::string srcpath;
    std::string savepath;
}Image;
// typedef cv::Mat Image;
typedef int Result;
// typedef struct result_s {
//     int boxnum;
//     void *boxes;
// } Result;

// #define Image int
class InferenceManager : public Logger {
protected:
    size_t poolSize;
    MemPool cpuMemoryPool;
    MemPoolGpu gpuMemoryPool;
    ThreadPool threadPool;
    YoloPool yoloPool;
    MySQLConnectionPool dbPool;  // MySQL 连接池
    // std::string modelName;
    // size_t buffersize = 1 << 24;
public:
    InferenceManager(size_t pool_size, size_t threads, const std::string& modelname);
    void processImages(const std::vector<Image>& images);
    // void processImages(const std::vector<cv::Mat>& images);
    Result processSingleImage(const Image& image);
    // Result processSingleImage(const cv::Mat& image);
    void handleResult(const Result& result);
};

#endif // INFERENCE_MANAGER_H
