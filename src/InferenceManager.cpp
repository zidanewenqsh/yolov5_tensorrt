#include "InferenceManager.h"
#include <cstddef>
#define dbHost "172.16.1.2"
#define dbUser "ai"
#define dbPass "12345678"
#define dbName "MyYoloDB"
InferenceManager::InferenceManager(size_t pool_size, size_t threads, const std::string& modelname)
    : poolSize(pool_size), cpuMemoryPool(pool_size, threads), 
        gpuMemoryPool(pool_size, threads), 
        threadPool(threads), yoloPool(modelname, threads),
        dbPool(dbHost, dbUser, dbPass, dbName, threads) {
    // YoloFactory::getInstance().setName(modelname);
    
    // 构造函数内容
    
    auto yolo = yoloPool.acquire();
    yolo->SetLevel(LogLevel::DEBUG);
    
    // 初始化
    if (yolo->build() < 0) {
        LOG_ERROR("build failed");
        // std::cout << "build failed" << std::endl;
        exit(-1);
    }
    yoloPool.release(yolo);
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


void InferenceManager::processBatch(const std::vector<Image>& images, int input_batch) {
    std::vector<std::future<Result>> results;
    std::vector<Image> batchImages;
    int j = 0;
    for (size_t i = 0; i < images.size(); i++) {
        if (j < input_batch && i < images.size()) {
            batchImages.push_back(images[i]);
            j++;
        }
        if (j == input_batch || i == images.size()){
            j = 0;
            results.emplace_back(
               threadPool.enqueue([this, batchImages]() {
                    return processBatchImage(batchImages);
                })
            );
            batchImages.clear();
        }
    }


    // 收集结果...
    for (auto& result : results) {
        // 处理每个结果
        handleResult(result.get());
    }
}
Result InferenceManager::processBatchImage(const std::vector<Image>& images) {
    // 从数据库连接池获取一个连接
    auto conn = dbPool.getConnection();

    // 从YoloPool获取一个Yolo实例
    auto yolo = yoloPool.acquire();
    auto yolov5 = dynamic_cast<Yolov5*>(yolo.get());
    yolov5->SetLevel(LogLevel::DEBUG);
    if (images.size() > static_cast<size_t>(yolov5->get_maxBatchSize())) {
        LOG_ERROR("batch num error");
    }

    // 为整个批次分配内存
    void *cpuYoloMemory = cpuMemoryPool.allocate();
    void *gpuYoloMemory = gpuMemoryPool.allocate();

    yolov5->setMemory(cpuYoloMemory, gpuYoloMemory, poolSize);

    // 准备存储处理后的图像数据
    // std::vector<std::shared_ptr<ImageData_t>> batchData;
    std::vector<ImageData_t> batchData;
    // batchData.reserve(images.size());

    // 处理每张图像并转换为 NCHW 格式
    for (const auto& image : images) {
        cv::Mat img = image.img;
        std::string imgname = image.imgname;
        std::string srcpath = image.srcpath;
        std::string savepath = image.savepath;
        std::shared_ptr<ImageData_t> imgdata = std::make_shared<ImageData_t>();
        // ImageData_t* imgdata = dynamic_cast<ImageData_t*>(data.get());
        yolov5->make_imagedata(img, savepath, imgdata.get());
        batchData.push_back(*imgdata);
    }

    // 执行批量推理
    yolov5->inferenceBatch(batchData);
    yolov5->drawimgBatch(batchData);
    // for (auto image:images) {
    //     yolov5->drawimg(image.img, image.savepath);
    // }
    // 处理推理结果...
    // 例如：保存结果、绘制边界框等

    // 推理完成后释放内存
    yolov5->reset();
    
    // 推理完成后释放内存
    cpuMemoryPool.deallocate(cpuYoloMemory);
    gpuMemoryPool.deallocate(gpuYoloMemory);

    // 将Yolo实例返回到池中
    yoloPool.release(std::move(yolo));
    dbPool.releaseConnection(conn);
    Result result = 0;
    return result;

}

Result InferenceManager::processSingleImage(const Image& image) {
    auto conn = dbPool.getConnection();
    // 从YoloPool获取一个Yolo实例
    auto yolo = yoloPool.acquire();
    auto yolov5 = dynamic_cast<Yolov5*>(yolo.get());
    yolov5->SetLevel(LogLevel::DEBUG);
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
    // int a = 1;
    // memcpy(cpuYoloMemory, &a, sizeof(int));

    std::shared_ptr<Data> data = std::make_shared<ImageData_t>();
    ImageData_t* imgdata = dynamic_cast<ImageData_t*>(data.get());
    yolov5->setMemory(cpuYoloMemory, gpuYoloMemory, poolSize);
    yolov5->make_imagedata(img, savepath, imgdata);

    // 使用Yolo实例进行推理
    yolo->inference(data.get());
    // printf("Result:%d\n", result);
    yolov5->drawimg(img, savepath);
    // Result result = 0;

    yolov5->reset();
    
    // 推理完成后释放内存
    cpuMemoryPool.deallocate(cpuYoloMemory);
    gpuMemoryPool.deallocate(gpuYoloMemory);

    // 将Yolo实例返回到池中
    yoloPool.release(std::move(yolo));
    dbPool.releaseConnection(conn);
    Result result = 0;
    return result;
}

void InferenceManager::handleResult(const Result& result) {
    // 处理推理结果的逻辑
}