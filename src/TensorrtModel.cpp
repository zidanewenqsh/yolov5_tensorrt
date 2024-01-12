#include "TensorrtModel.h"
// #include "logger.h"
// #include "cuda_utils.h"
#include "Logger.h"
#include "YoloUtils.h"
// #include "yolox_config.h"
#define PROCESSONGPU 1
#define USEOPENCV 1

bool exists(const std::string& path){

#ifdef _WIN32
    return ::PathFileExistsA(path.c_str());
#else
    return access(path.c_str(), F_OK) == 0;
#endif
}

#define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)
static bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line){
    if(code != cudaSuccess){    
        const char* err_name = cudaGetErrorName(code);    
        const char* err_message = cudaGetErrorString(code);  
        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);   
        return false;
    }
    return true;
}
class cudaLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char *msg) noexcept override {
        if (severity != Severity::kINFO) {
            std::cout << msg << std::endl;
        }
    }
} static gLogger;

// static int totalsize = 0;
// static int idx = 0;
TensorRTModel::TensorRTModel() {
    // std::cout << "TensorRTModel()" << std::endl;
    LOG_INFO("TensorRTModel()");
    cudaStreamCreate(&stream);
    
}
TensorRTModel::TensorRTModel(const std::string &name){
    LOG_INFO("TensorRTModel()");
    // std::cout << "TensorRTModel()" << std::endl;
    cudaStreamCreate(&stream);
    onnx_file = name + ".onnx";
    trt_file = name + ".trt";
    // std::cout << "trt_file:" << trt_file << std::endl;
    // std::cout << "TensorRTModel() init finished" << std::endl;
}
TensorRTModel::TensorRTModel(const std::string &name, int buffer_size):buffersize(buffer_size) {
    // std::cout << "TensorRTModel()" << std::endl;
    // checkRuntime(cudaMallocHost(&host_buffer, buffer_size));
    // checkRuntime(cudaMalloc(&device_buffer, buffer_size));
    cudaStreamCreate(&stream);
    host_buffer_now = (char*)host_buffer;
    device_buffer_now = (char*)device_buffer;
    onnx_file = name + ".onnx";
    trt_file = name + ".trt";
    // std::cout << "trt_file:" << trt_file << std::endl;
    // std::cout << "TensorRTModel() init finished" << std::endl;
}
TensorRTModel::~TensorRTModel(){
    LOG_INFO("~TensorRTModel()");
    // std::cout << "~TensorRTModel()" << std::endl;
    // checkRuntime(cudaFreeHost(host_buffer));
    // checkRuntime(cudaFree(device_buffer));
    checkRuntime(cudaStreamDestroy(stream));
}

int TensorRTModel::malloc_host(void **ptr, size_t size) {
    if (*ptr != nullptr) {
        std::cout << "malloc_host failed, *ptr is not nullptr" << std::endl;
        return -1;
    }
    if (host_buffer_now + size > (char*)host_buffer + buffersize) {
        std::cout << "malloc_host failed, size is not enough" << std::endl;
        return -1;
    }
    *ptr = host_buffer_now;
    host_buffer_now += size;
    // totalsize += size;
    return 0; 
}
int TensorRTModel::malloc_device(void** ptr, size_t size) {
    if (*ptr != nullptr) {
        std::cout << "malloc_device failed, *ptr is not nullptr" << std::endl;
        return -1;
    }
    if (device_buffer_now + size > (char*)device_buffer + buffersize) {
        std::cout << "malloc_device failed, size is not enough" << std::endl;
        return -1;
    }
    *ptr = device_buffer_now;
    device_buffer_now += size;
    return 0;
}