// #include "IYolo.h"
#include "Yolov5.h"
#include "Logger.h"
#include "YoloConfig.h"
#include "YoloUtils.h"
#include <iostream>
// #include "logger.h"
// #include "cuda_utils.h"
#define PROCESSONGPU 1
#define CPUNMS 1
#define USEOPENCV 0

// static int idx = 0;


class cudaLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char *msg) noexcept override {
        if (severity != Severity::kINFO) {
            std::cout << msg << std::endl;
        }
    }
} static gLogger;

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

Yolov5::Yolov5(const std::string& name) 
    : TensorRTModel(name){
    LOG_INFO("Yolov5()");
    runtime = nvinfer1::createInferRuntime(gLogger);
    // LOG_INFO("createInferRuntime");
    if (!runtime) {
        // std::cout << "createInferRuntime failed" << std::endl;
        LOG_FATAL("createInferRuntime failed");
        exit(-1);
    }
    auto engine_data = loadData(trt_file);
    engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
    if (!engine) {
        // std::cout << "deserializeCudaEngine failed" << std::endl;
        LOG_FATAL("deserializeCudaEngine failed");
        exit(-1);
    }
    if (engine->getNbIOTensors() != 2) {
        // std::cout << "getNbBindings failed" << std::endl;
        LOG_FATAL("getNbBindings failed");
        exit(-1);
    }
    context = engine->createExecutionContext();
    if (!context) {
        // std::cout << "createExecutionContext failed" << std::endl;
        LOG_FATAL("createExecutionContext failed");
        exit(-1);
    }
    // if (init() < 0) {
    //     std::cout << "init failed" << std::endl;
    //     exit(-1);
    // }
    // 初始化
    // if (build() < 0) {
    //     std::cout << "build failed" << std::endl;
    //     exit(-1); 
    // }
    // std::cout << "Yolov5()" << std::endl;
}

Yolov5::Yolov5(const std::string& name, int buffer_size) 
    : TensorRTModel(name, buffer_size){
    // if (build() < 0) {
    //     LOG_ERROR("build failed");
    //     // std::cout << "build failed" << std::endl;
    //     exit(-1); 
    // }
    // if (init() < 0) {
    //     LOG_ERROR("init failed");
    //     // std::cout << "init failed" << std::endl;
    //     exit(-1);
    // }
    // std::cout << "Yolov5()" << std::endl;
}

Yolov5::~Yolov5() {
    std::cout << "~Yolov5())" << std::endl;
}
int Yolov5::init() {
    if (!build_finished && !exists(trt_file)) {
        // std::cout << "trt_file exists" << std::endl;
        LOG_FATAL("trt file not exists: %s", trt_file.c_str());
        return -1;
    }
    
    checkRuntime(cudaMemset((char*)host_buffer, 0, buffersize));
    checkRuntime(cudaMemset((char*)device_buffer, 0, buffersize));
    // LOG_INFO("temp:%d", temp);
    // temp = 100;
    inputW = kInputW;
    inputH = kInputH;
    if (host_buffer == nullptr || device_buffer == nullptr) {
        std::cout << "host_buffer or device_buffer is nullptr" << std::endl;
        return -1;
    }
    // 初始化
    // if (build() < 0) {
    //     std::cout << "build failed" << std::endl;
    //     return -1;
    // }
    // LOG_INFO("init_finished value:%d\n", init_finished);
    if (init_finished) {
        std::cout << "runtime_init already finished" << std::endl;
        return 0;
    }
    float mean[3] = {0, 0, 0};
    float std[3] = {1, 1, 1};
    // if (runtime == nullptr) {
    //     runtime = nvinfer1::createInferRuntime(gLogger);
    //     LOG_INFO("createInferRuntime");
    // }
    // if (!runtime) {
    //     std::cout << "createInferRuntime failed" << std::endl;
    //     return -1;
    // }
    // auto engine_data = loadData(trt_file);
    // engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
    // if (!engine) {
    //     std::cout << "deserializeCudaEngine failed" << std::endl;
    //     return -1;
    // }
    // if (engine->getNbIOTensors() != 2) {
    //     std::cout << "getNbBindings failed" << std::endl;
    //     return -1;
    // }
    // context = engine->createExecutionContext();
    // if (!context) {
    //     std::cout << "createExecutionContext failed" << std::endl;
    //     return -1;
    // }

    // 明确当前推理时，使用的数据输入大小
    auto input_dims = engine->getTensorShape(kInputTensorName);
    input_batch = input_dims.d[0];
    input_numel = input_batch * kInputH * kInputW * 3;
    if(malloc_host(&input_data_host, sizeof(float) * input_numel)<0) {
        std::cout << "malloc_host input_data_host failed" << std::endl;
        return -1;
    }
    if (malloc_device(&input_data_device, sizeof(float) * input_numel)<0){
        std::cout << "malloc_device input_data_device failed" << std::endl;
        return -1;
    }

    // 明确当前推理时，使用的数据输出大小
    nvinfer1::Dims output_dims = context->getTensorShape(kOutputTensorName);
    // printDims(output_dims);
    output_batch = output_dims.d[0];
    output_numbox = output_dims.d[1];
    output_numprob = output_dims.d[2];
    num_classes = output_numprob - 5;
    output_numel = output_batch * output_numbox * output_numprob;
    // printf("output_batch: %d, output_numbox: %d, output_numprob: %d, num_classes: %d, output_numel: %d\n", output_batch, output_numbox, output_numprob, num_classes, output_numel); 

    if(malloc_host(&output_data_host, sizeof(float) * output_numel)<0) {
        std::cout << "malloc_host output_data_host failed" << std::endl;
        return -1;
    }
    if (malloc_device(&output_data_device, sizeof(float) * output_numel)<0){
        std::cout << "malloc_device output_data_device failed" << std::endl;
        return -1;
    }

    if (malloc_host(&h_mean, 3 * sizeof(float))<0) {
        std::cout << "malloc_device h_mean failed" << std::endl;
        return -1;
    }
    // printf("h_mean:%p\n", h_mean);
    // h_mean = new float[3];
    memcpy(h_mean, mean, 3 * sizeof(float));
    // checkRuntime(cudaMemcpy(h_mean, mean, 3 * sizeof(float), cudaMemcpyHostToHost));
    if (malloc_host(&h_std, 3 * sizeof(float))<0) {
        std::cout << "malloc_device h_std failed" << std::endl;
        return -1;
    }
    // h_std = new float[3];
    memcpy(h_std, std, 3 * sizeof(float));
    if (malloc_device(&d_mean, 3 * sizeof(float))<0) {
        std::cout << "malloc_device d_mean failed" << std::endl;
        return -1;
    }
    if (malloc_device(&d_std, 3 * sizeof(float))<0) {
        std::cout << "malloc_device d_std failed" << std::endl;
        return -1;
    }
    checkRuntime(cudaMemcpyAsync(d_mean, mean, 3 * sizeof(float), cudaMemcpyHostToDevice, stream));
    checkRuntime(cudaMemcpyAsync(d_std, std, 3 * sizeof(float), cudaMemcpyHostToDevice, stream));

    if (malloc_host(&h_matrix, 12 * sizeof(float))<0) {
        std::cout << "malloc_device h_matrix failed" << std::endl;
        return -1;
    }

    if (malloc_device(&d_matrix, 12 * sizeof(float))<0) {
        std::cout << "malloc_device d_matrix failed" << std::endl;
        return -1;
    }

    // if (malloc_device(&image_data_device, kChannel * kImageHMax * kImageWMax * sizeof(unsigned char))<0) {
    //     std::cout << "malloc_device image_data_device failed" << std::endl;
    //     return -1;
    // }

    if (malloc_host(&h_filtered_boxes, sizeof(int) * (output_numel + 1))) {
        std::cout << "malloc_host h_filtered_boxes failed" << std::endl;
        return -1;
    }
    if (malloc_device(&d_filtered_boxes, sizeof(int) * (output_numel) + 1)) {
        std::cout << "malloc_device d_filtered_boxes failed" << std::endl;
        return -1;
    }
    // if (malloc_host(&h_filtered_boxes, (sizeof(gBox) + sizeof(int)) * output_numbox)) {
    //     std::cout << "malloc_host h_filtered_boxes failed" << std::endl;
    //     return -1;
    // }
    // if (malloc_device(&d_filtered_boxes, (sizeof(gBox) + sizeof(int)) * output_numbox)) {
    //     std::cout << "malloc_device d_filtered_boxes failed" << std::endl;
    //     return -1;
    // }
    if (malloc_host(&h_box_count, sizeof(int) * output_batch)) {
        std::cout << "malloc_host h_box_count failed" << std::endl;
        return -1;
    }
    if (malloc_device(&d_box_count, sizeof(int) * output_batch)) {
        std::cout << "malloc_device d_box_count failed" << std::endl;
        return -1;
    }

    init_finished = true;
    // std::cout << "init finished" << std::endl;
    LOG_INFO("init finieshed");
    return 0;

}


int Yolov5::preprocess(ImageData_t *imgdata) {
    if (malloc_device(&image_data_device, imgdata->numel * sizeof(unsigned char))<0) {
        std::cout << "malloc_device image_data_device failed" << std::endl;
        return -1;
    }
    // 前处理
    // std::cout << "preprocess" << std::endl;
    calculate_matrix((float*)h_matrix, imgdata->width, imgdata->height, kInputW, kInputH);
#if PROCESSONGPU
// #if 0
    float *d2i = (float*)h_matrix + 6;
    checkRuntime(cudaMemcpyAsync(d_matrix, d2i, 6 * sizeof(float), cudaMemcpyHostToDevice, stream));
    checkRuntime(cudaMemset((char*)d_box_count, 0, sizeof(int)));
    // checkRuntime(cudaMemset((char*)d_filtered_boxes, 0, output_numbox * sizeof(gBox)));
    checkRuntime(cudaMemset((char*)d_filtered_boxes, 0, (output_numel + 1) * sizeof(int)));
    checkRuntime(cudaMemcpyAsync(image_data_device, imgdata->dataptr, imgdata->size, cudaMemcpyHostToDevice, stream));
    checkRuntime(cudaStreamSynchronize(stream));
    // std::cout << imgdata->width << std::endl;
    int ret = preprocess_gpu((unsigned char*)image_data_device, (float*)input_data_device, 
        imgdata->width, imgdata->height, kInputW, kInputH, 
        (float*)d_matrix, (float*)d_mean, (float*)d_std);
    if (ret < 0) {
        // std::cout << "preprocess failed" << std::endl;
        LOG_ERROR("preprocess not success");
        return -1;
    }
#else
#if USEOPENCV
    float *i2d = (float*)h_matrix;
    preprocess_opencv_cpu(img, (float*)input_data_host, i2d, kInputW, kInputH, (float *)h_mean, (float*)h_std);
#else
    float *d2i = (float*)h_matrix + 6;
    preprocess_cpu_v2(img, (float*)input_data_host, d2i, kInputW, kInputH, false);

#endif
    checkRuntime(cudaMemcpyAsync(input_data_device, input_data_host, input_numel * sizeof(float), cudaMemcpyHostToDevice, stream));
    checkRuntime(cudaStreamSynchronize(stream));
#endif
    return 0;
}


int Yolov5::preprocess(cv::Mat &img) {
    // 前处理
    // std::cout << "preprocess" << std::endl;
    calculate_matrix((float*)h_matrix, img.cols, img.rows, kInputW, kInputH);
#if PROCESSONGPU
// #if 0
    float *d2i = (float*)h_matrix + 6;
    checkRuntime(cudaMemcpyAsync(d_matrix, d2i, 6 * sizeof(float), cudaMemcpyHostToDevice, stream));
    checkRuntime(cudaMemset((char*)d_box_count, 0, sizeof(int)));
    // checkRuntime(cudaMemset((char*)d_filtered_boxes, 0, output_numbox * sizeof(gBox)));
    checkRuntime(cudaMemset((char*)d_filtered_boxes, 0, (output_numel + 1) * sizeof(int)));
    checkRuntime(cudaMemcpyAsync(image_data_device, img.data, img.total() * img.channels(), cudaMemcpyHostToDevice, stream));
    checkRuntime(cudaStreamSynchronize(stream));
    int ret = preprocess_gpu((unsigned char*)image_data_device, (float*)input_data_device, 
        img.cols, img.rows, kInputW, kInputH, 
        (float*)d_matrix, (float*)d_mean, (float*)d_std);
    if (ret < 0) {
        std::cout << "preprocess failed" << std::endl;
        return -1;
    }
#else
#if USEOPENCV
    float *i2d = (float*)h_matrix;
    preprocess_opencv_cpu(img, (float*)input_data_host, i2d, kInputW, kInputH, (float *)h_mean, (float*)h_std);
#else
    float *d2i = (float*)h_matrix + 6;
    preprocess_cpu_v2(img, (float*)input_data_host, d2i, kInputW, kInputH, false);

#endif
    checkRuntime(cudaMemcpyAsync(input_data_device, input_data_host, input_numel * sizeof(float), cudaMemcpyHostToDevice, stream));
    checkRuntime(cudaStreamSynchronize(stream));
#endif
    return 0;
}
int Yolov5::postprocess() {
    // 后处理
    // std::cout << "postprocess()" << std::endl;
#if PROCESSONGPU
    cudaMemcpy(output_data_host, output_data_device, output_numel * sizeof(float), cudaMemcpyDeviceToHost);
    // saveWeight("yolov5output2.data", (float*)output_data_host, output_numel);

    // 调用封装的函数
    bool cpunms = false;
#if CPUNMS
    cpunms = true;
#endif
    int ret = postprocess_cuda((float*)output_data_device, (char *)d_filtered_boxes, (float *)d_matrix, 
             output_batch, output_numbox, output_numprob, kConfThresh, kNmsThresh, cpunms);
    if (ret < 0) {
        std::cout << "postprocess failed" << std::endl;
        return -1;
    } 
    // // 在后处理的函数内部，已经完成了h_box_count的同步
    // checkRuntime(cudaMemcpyAsync(h_box_count, d_box_count, sizeof(int), cudaMemcpyDeviceToHost, stream));
    // checkRuntime(cudaStreamSynchronize(stream)); // 我要用h_box_count，所以要同步
    // checkRuntime(cudaMemcpyAsync(h_box_count, (char *)d_filtered_boxes, sizeof(int), cudaMemcpyDeviceToHost, stream));
    // checkRuntime(cudaMemcpyAsync(h_box_count, (char *)d_filtered_boxes, sizeof(int), cudaMemcpyDeviceToHost, stream));
    checkRuntime(cudaMemcpyAsync(h_filtered_boxes, (char*)d_filtered_boxes, (output_numel + 1) * sizeof(int), cudaMemcpyDeviceToHost, stream));
    checkRuntime(cudaStreamSynchronize(stream)); // 我要用h_box_count，所以要同步
    // for (int i = 0; i < output_batch; i++) {
    //     printf("h_box_count[%d]: %d\n", i, *((int*)h_box_count + i));
    // }
    // checkRuntime(cudaMemcpyAsync(h_filtered_boxes, (char*)d_filtered_boxes + sizeof(int), (*((int*)h_box_count)) + sizeof(int), cudaMemcpyDeviceToHost, stream));
    // checkRuntime(cudaMemcpyAsync(h_filtered_boxes, d_filtered_boxes, (*((int*)h_box_count)) * sizeof(gBox), cudaMemcpyDeviceToHost, stream));
    checkRuntime(cudaStreamSynchronize(stream));
#if CPUNMS
    nms_cpu((char*)h_filtered_boxes, kNmsThresh);
#endif
#else
    checkRuntime(cudaMemcpyAsync(output_data_host, output_data_device, sizeof(float) * output_numel, cudaMemcpyDeviceToHost, stream));
    checkRuntime(cudaStreamSynchronize(stream));
#endif
    return 0;
}
void Yolov5::drawimg(cv::Mat &img, const std::string& savepath) {
    // 绘制
    // std::cout << "draw()" << std::endl;
#if PROCESSONGPU
    // int count = *(int *)h_box_count;
    // LogInfo("count: %d\n", count);
    // LOG_INFO("count: %d\n", count);
    // printf("count: %d\n", count);
    cv::Mat img_draw = draw_gpu((char*)h_filtered_boxes, img);
    // cv::imwrite("result_gpu_v2.jpg", img_draw);
    // cv::imwrite("result_gpu_" + std::to_string(index) + ".jpg", img_draw);
    cv::imwrite(savepath, img_draw);
    // printf("image %s saved success\n", savepath.c_str());
    // index++;
#else
    float *d2i = (float*)h_matrix + 6;
    std::vector<Box> resboxes = postprocess_cpu((float*)output_data_host, output_batch, output_numbox, output_numprob, kConfThresh, kNmsThresh);
    // 绘制结果
    cv::Mat img_draw = draw_cpu(resboxes, img, d2i);
    cv::imwrite(savepath, img_draw);
    printf("image %s saved success\n", savepath.c_str());
#endif
}
int Yolov5::forward() {
    // 推理
    // std::cout << "inference()" << std::endl;
    void* bindings[] = {input_data_device, output_data_device};
    bool success = context->executeV2(bindings);
    if (!success) {
        std::cout << "enqueue failed" << std::endl;
        return -1;
    }
    return 0;
}

int Yolov5::inference(Data *data) { 
    if (!set_memory) {
        LOG_ERROR("not set memory");
        // std::cout << "set_memory is false" << std::endl;
        return -1;
    }
    init();
    // if (!init_finished) {
    //     // std::cout << "not init" << std::endl;
    //     init();
    //     // return -1;
    // }
    // ImageData_t* imageData = dynamic_cast<ImageData_t*>(data);
    ImageData_t* imagedata = static_cast<ImageData_t*>(data);
    
    // if (imageData) {
    //     // 现在可以安全地使用 imageData 中的特定字段
    //     // ...
    //     return 0; // 或其他适当的返回值
    // } else {
    //     // 转换失败的处理
    //     return -1; // 或其他错误代码
    // }
    // int size = imagedata->size;
    // void *dataptr = imagedata->dataptr;
    // if(malloc_host(&input_data_host, sizeof(float) * input_numel)<0) {
    //     std::cout << "malloc_host input_data_host failed" << std::endl;
    //     return -1;
    // }
    // if (malloc_device(&input_data_device, sizeof(float) * input_numel)<0){
    //     std::cout << "malloc_device input_data_device failed" << std::endl;
    //     return -1;
    // }
    int ret;
    auto start = std::chrono::high_resolution_clock::now();
    ret = preprocess(imagedata);
    if (ret < 0) {
        // std::cout << "preprocess failed" << std::endl;
        LOG_ERROR("preprocess failed");
        return -1;
    }
    // printf("input_data_device:%p\n", input_data_device);
    // printf("output_data_device:%p\n", output_data_device);
    ret = forward();
    if (ret < 0) {
        std::cout << "inference failed" << std::endl;
        return -1;
    }
    ret = postprocess();
    if (ret < 0) {
        std::cout << "postprocess failed" << std::endl;
        return -1;
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    // std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;
    LOG_INFO("Elapsed time:%f\n", elapsed.count());
    return 0;
}
int Yolov5::get_box(void **boxes) {
    int count = *(int *)h_filtered_boxes;
    *boxes = (char*)h_filtered_boxes + sizeof(int); 
    return count;
}
int Yolov5::inference_image(std::string &img_file) {
    if (malloc_host(&h_filtered_boxes, sizeof(int) * (output_numel + 1))) {
        std::cout << "malloc_host h_filtered_boxes failed" << std::endl;
        return -1;
    }
    if (malloc_device(&d_filtered_boxes, sizeof(int) * (output_numel) + 1)) {
        std::cout << "malloc_device d_filtered_boxes failed" << std::endl;
        return -1;
    }
    // auto imgname = extractFilenameWithoutExtension(img_file);
    auto imgname = extractFilename(img_file);
    printf("imgname:%s\n", imgname.c_str());
    std::string savepath = "result_" + imgname; 
    printf("savepath:%s\n", savepath.c_str());
    // 处理一张图片
    cv::Mat img = cv::imread(img_file);
    if (img.empty()) {
        printf("cv::imread %s failed\n", img_file.c_str());
        return -1;
    }
    int ret;
    auto start = std::chrono::high_resolution_clock::now();
    ret = preprocess(img);
    if (ret < 0) {
        // std::cout << "preprocess failed" << std::endl;
        LOG_ERROR("preprocess error");
        return -1;
    }
    ret = forward();
    if (ret < 0) {
        // std::cout << "inference failed" << std::endl;
        LOG_ERROR("inference error");
        return -1;
    }
    ret = postprocess();
    if (ret < 0) {
        // std::cout << "postprocess failed" << std::endl;
        LOG_ERROR("postprocess error");
        return -1;
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;
    drawimg(img, savepath);
    // free_host(h_filtered_boxes);
    // free_device(d_filtered_boxes);
    return 0;
}
int Yolov5::setMemory(void *cpuptr, void *gpuptr, int buffer_size) {
    if (cpuptr == nullptr || gpuptr == nullptr|| buffer_size <= 0) {
        // std::cout << "setMemory failed" << std::endl;
        LOG_ERROR("setMemory failed");
        return -1;
    }
    host_buffer = cpuptr;
    device_buffer = gpuptr;
    host_buffer_now = (char*)host_buffer;
    device_buffer_now = (char*)device_buffer;
    buffersize = buffer_size;
    set_memory = true;
    return 0;
}
void Yolov5::make_imagedata(const cv::Mat& image, ImageData_t* imagedata) {
    imagedata->width = image.cols;
    imagedata->height = image.rows;
    imagedata->channels = image.channels();
    imagedata->batch = 1;
    imagedata->numel = image.total() * image.channels();
    imagedata->dataptr = image.data;
    imagedata->size = image.total() * image.elemSize();
}



// int Yolov5::malloc_host(void **ptr, size_t size) {
//     if (mempool == nullptr) {
//         mempool = std::make_unique<MemoryPool>();
//         if (mempool == nullptr) {
//             std::cerr << "mempool is nullptr" << std::endl;
//             return -1;
//         }
//         // std::cerr << "mempool is nullptr" << std::endl;
//         // return -1;
//     }
//     *ptr =  mempool->allocate(size);
//     return 0; 
// }
// int Yolov5::malloc_device(void** ptr, size_t size) {
//     if (mempool_gpu == nullptr) {
//         mempool_gpu = std::make_unique<MemoryPoolGpu>();
//         if (mempool_gpu == nullptr) {
//             std::cerr << "mempool_gpu is nullptr" << std::endl;
//             return -1;
//         }
//         // std::cerr << "mempool is nullptr" << std::endl;
//         // return -1;
//     }
//     *ptr =  mempool_gpu->allocate(size);
//     return 0;
// }
// int Yolov5::free_host(void *ptr) {
//     if (mempool == nullptr) {
//         std::cerr << "mempool is nullptr" << std::endl;
//         return -1;
//     }
//     mempool->deallocate(ptr);
//     std::cout << "free_device success" << std::endl;
//     return 0; 
// }
// int Yolov5::free_device(void* ptr) {
//     if (mempool_gpu == nullptr) {
//         std::cerr << "mempool is nullptr" << std::endl;
//         return -1;
//     }
//     mempool_gpu->deallocate(ptr);
//     std::cout << "free_device success" << std::endl;
//     return 0;
// }
int Yolov5::build() {
    // 生成engine
    /* 这里面为何用智能指针，是因为如果不用构建，这些变量都用不到
    */
    if (exists(trt_file)) {
        build_finished = true;
        // std::cout << "trt_file exists" << std::endl;
        LOG_INFO("trt file exists: %s", trt_file.c_str());
        return 0;
    }
    if (!exists(onnx_file)) {
        // std::cout << "onnx_file not exists" << std::endl;
        LOG_ERROR("onnx file not exists: %s", onnx_file.c_str());
        return -1;
    }
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    if (!builder) {
        // std::cout << "createInferBuilder failed" << std::endl;
        LOG_ERROR("createInferBuilder failed");
        return -1;
    }
    auto explictBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);  // NOLINT   
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explictBatch));
    if (!network) {
        // std::cout << "createNetworkV2 failed" << std::endl;
        LOG_ERROR("createNetworkV2 failed");
        return -1;
    }
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    if (!parser) {
        // std::cout << "createParser failed" << std::endl;
        LOG_ERROR("createParser failed");
        return -1;
    }
    // 创建引擎
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        // std::cout << "createBuilderConfig failed" << std::endl;
        LOG_ERROR("createBuilderConfig failed");
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
        // std::cout << "parseFromFile failed" << std::endl;
        LOG_ERROR("parseFromFile failed");
        return -1;
    }
    auto input = network->getInput(0);
    nvinfer1::IOptimizationProfile *profile = builder->createOptimizationProfile();

    if (!profile) {
        // std::cout << "createOptimizationProfile failed" << std::endl;
        LOG_ERROR("createOptimizationProfile failed");
        return -1;
    }
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, channelSize, inputH, inputW});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{optBatchSize, channelSize, inputH, inputW});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{maxBatchSize, channelSize, inputH, inputW});
    config->addOptimizationProfile(profile);
    auto serialized_engine = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!serialized_engine) {
        std::cout << "buildSerializedNetwork failed" << std::endl;
        LOG_ERROR("createInferBuilder failed");
        return -1;
    }
    saveData(trt_file, reinterpret_cast<char*>(serialized_engine->data()), serialized_engine->size());
    // printf("Serialize engine success\n");
    LOG_INFO("Serialize engine success");
    build_finished = true;
    return 0;

}
void Yolov5::reset() {
    host_buffer=nullptr;
    device_buffer=nullptr;
    host_buffer_now=nullptr;
    device_buffer_now=nullptr;
    image_data_device = nullptr;
    input_data_host = nullptr;
    input_data_device = nullptr;
    output_data_host = nullptr;
    output_data_device = nullptr;
    h_mean = nullptr;
    h_std = nullptr;
    d_mean = nullptr;
    d_std = nullptr;
    h_matrix = nullptr;
    d_matrix = nullptr;
    h_filtered_boxes = nullptr;
    d_filtered_boxes = nullptr;
    h_box_count = nullptr;
    d_box_count = nullptr;
    init_finished = false;
    use_int8 = false;
    use_fp16 = false;
    set_memory = false;
    LOG_INFO("reset finished");
}