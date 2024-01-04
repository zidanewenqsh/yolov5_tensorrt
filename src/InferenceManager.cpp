#include "InferenceManager.h"

InferenceManager::InferenceManager(size_t pool_size, size_t threads, std::string& modelname)
    : poolSize(pool_size), cpuMemoryPool(pool_size), gpuMemoryPool(pool_size), 
        threadPool(threads), yoloPool(modelname) {
    // 构造函数内容
}

void InferenceManager::processImages(const std::vector<Image>& images) {
    std::vector<std::future<Result>> results;

    for (const auto& image : images) {
        results.emplace_back(
            threadPool.enqueue([this, &image]() {
                return processSingleImage(image);
            })
        );
    }

    // 收集结果...
    for (auto& result : results) {
        // 处理每个结果
        handleResult(result.get());
    }
}

Result InferenceManager::processSingleImage(const Image& image) {
    // 从YoloPool获取一个Yolo实例
    auto yolo = yoloPool.acquire();

    // 从内存池中分配内存给Yolo实例
    // void* yoloMemory = memoryPool.allocate();
    // size_t size = 1 << 25;
    // MemPool cpumem(size);
    // MemPoolGpu gpumem(size);

    // void *cpuptr = cpumem.allocate();
    // void *gpuptr = gpumem.allocate();

    cv::Mat img = image.img;
    std::string imgname = image.imgname;
    std::string srcpath = image.srcpath;
    std::string savepath = image.savepath;

    void *cpuYoloMemory = cpuMemoryPool.allocate();
    void *gpuYoloMemory = gpuMemoryPool.allocate();


    std::shared_ptr<Data> data = std::make_shared<ImageData_t>();
    ImageData_t* imgdata = dynamic_cast<ImageData_t*>(data.get());
    yolo->setMemory(cpuYoloMemory, gpuYoloMemory, poolSize);
    yolo->make_imagedata(img, imgdata);
    // 使用Yolo实例进行推理
    Result result = yolo->inference(data.get());
    // printf("Result:%d\n", result);
    yolo->drawimg(img, savepath);
    // Result result = 0;

    // 推理完成后释放内存
    cpuMemoryPool.deallocate(cpuYoloMemory);
    gpuMemoryPool.deallocate(gpuYoloMemory);

    // 将Yolo实例返回到池中
    yoloPool.release(std::move(yolo));

    return result;
}

void InferenceManager::handleResult(const Result& result) {
    // 处理推理结果的逻辑
}
