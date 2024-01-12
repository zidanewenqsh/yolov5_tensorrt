#ifndef __YOLO_H__
#define __YOLO_H__

#include "TensorrtModel.h"
#include "Logger.h"
// #include "yolov5_config.h"
#include "preprocess_gpu.h"
#include "postprocess_gpu.h"
#include <cstdlib>
#include <chrono>
#include "preprocess.h"
#include "postprocess.h"
// #include "IYolo.h"
// #include "yolov5_utils.h"
// #include "mempool.h"
// #include "mempool_gpu.h"

typedef struct ImageData_s:Data {
    int width;
    int height;
    int channels;
    int batch;
    int numel;
    std::string savepath;
}ImageData_t;

typedef struct matrix_s {
    float i2d[6];
    float d2i[6];
} Matrix_t;

// typedef struct inputData_s {
    
// } InputData_t;

class Yolov5:public TensorRTModel {
public:
    Yolov5();
    Yolov5(const std::string& name);
    Yolov5(const std::string& name, int buffer_size);
    ~Yolov5();
    int build() override;
    int init() override;
    int init(const std::vector<ImageData_t*> datas);
    void reset() override;
    int setMemory(void *cpuptr, void *gpuptr, int buffersize) override;
    int preprocess(cv::Mat &img) override;
    int preprocess(ImageData_t *imgdata);
    int preprocessBatch(std::vector<ImageData_t>& imgdatas);
    int postprocess() override;
    int postprocessBatch();
    void drawimg(cv::Mat &img, const std::string& savepath);
    void drawimgBatch(std::vector<ImageData_t>& imgdatas);
    int forward() override;
    int inference(Data* data) override;
    int inferenceBatch(std::vector<ImageData_t>& datas);
    int inference_image(std::string &img_file);
    void make_imagedata(const cv::Mat& image, const std::string& savepath, ImageData_t* imagedata);
    int get_box(void **boxes);
    int get_maxBatchSize();
private:
    int maxBatchSize = 10;
    nvinfer1::Dims inputDims;
    nvinfer1::Dims outputDims;
    // int malloc_host(void **ptr, size_t size);
    // int malloc_device(void** ptr, size_t size);
    // int free_host(void *ptr);
    // int free_device(void * ptr);
// private:
    // MemPool mempool;
    // MemPoolGpu mempool_gpu;
// private:
//     std::unique_ptr<IYolo> createYoloInstance() override;
//     std::unique_ptr<MemoryPool> mempool;
//     std::unique_ptr<MemoryPoolGpu> mempool_gpu;
};

#endif // __YOLO_H_