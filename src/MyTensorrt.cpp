#include "MyTensorrt.h"
// #include "logger.h"
// #include "cuda_utils.h"
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
MyTensorRT::MyTensorRT() {
    std::cout << "MyTensorRT()" << std::endl;
    cudaStreamCreate(&stream);
    
}
MyTensorRT::MyTensorRT(std::string &name){
    std::cout << "MyTensorRT()" << std::endl;
    cudaStreamCreate(&stream);
    onnx_file = name + ".onnx";
    trt_file = name + ".trt";
    std::cout << "trt_file:" << trt_file << std::endl;
    std::cout << "MyTensorRT() init finished" << std::endl;
}
MyTensorRT::MyTensorRT(std::string &name, int buffer_size):buffersize(buffer_size) {
    std::cout << "MyTensorRT()" << std::endl;
    // checkRuntime(cudaMallocHost(&host_buffer, buffer_size));
    // checkRuntime(cudaMalloc(&device_buffer, buffer_size));
    cudaStreamCreate(&stream);
    host_buffer_now = (char*)host_buffer;
    device_buffer_now = (char*)device_buffer;
    onnx_file = name + ".onnx";
    trt_file = name + ".trt";
    std::cout << "trt_file:" << trt_file << std::endl;
    std::cout << "MyTensorRT() init finished" << std::endl;
}
MyTensorRT::~MyTensorRT(){
    std::cout << "~MyTensorRT()" << std::endl;
    // checkRuntime(cudaFreeHost(host_buffer));
    // checkRuntime(cudaFree(device_buffer));
    checkRuntime(cudaStreamDestroy(stream));
}

int MyTensorRT::build() {
    // 生成engine
    /* 这里面为何用智能指针，是因为如果不用构建，这些变量都用不到
    */
    if (exists(trt_file)) {
        std::cout << "trt_file exists" << std::endl;
        return 0;
    }
    if (!exists(onnx_file)) {
        std::cout << "onnx_file not exists" << std::endl;
        return -1;
    }
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    if (!builder) {
        std::cout << "createInferBuilder failed" << std::endl;
        return -1;
    }
    auto explictBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);  // NOLINT   
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explictBatch));
    if (!network) {
        std::cout << "createNetworkV2 failed" << std::endl;
        return -1;
    }
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    if (!parser) {
        std::cout << "createParser failed" << std::endl;
        return -1;
    }
    // 创建引擎
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        std::cout << "createBuilderConfig failed" << std::endl;
        return -1;
    }

    // builder->setMaxBatchSize(1);
    if (use_fp16) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    // config->setFlag(nvinfer1::BuilderFlag::);
    // 设置工作区的大小
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1<<28);
    
    // 创建onnxparser
    auto success= parser->parseFromFile(onnx_file.c_str(), 1);
    if (!success) {
        std::cout << "parseFromFile failed" << std::endl;
        return -1;
    }
    auto input = network->getInput(0);
    nvinfer1::IOptimizationProfile *profile = builder->createOptimizationProfile();

    if (!profile) {
        std::cout << "createOptimizationProfile failed" << std::endl;
        return -1;
    }
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, channelSize, inputH, inputW});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{optBatchSize, channelSize, inputH, inputW});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{maxBatchSize, channelSize, inputH, inputW});
    config->addOptimizationProfile(profile);
    auto serialized_engine = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!serialized_engine) {
        std::cout << "buildSerializedNetwork failed" << std::endl;
        return -1;
    }
    saveData(trt_file, reinterpret_cast<char*>(serialized_engine->data()), serialized_engine->size());
    printf("Serialize engine success\n");
    return 0;

}
int MyTensorRT::malloc_host(void **ptr, size_t size) {
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
int MyTensorRT::malloc_device(void** ptr, size_t size) {
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