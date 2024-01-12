#include "YoloPool.h"
#include "TensorrtModel.h"
#include "YoloFactory.h"
#include "Logger.h"
#include <cstddef>

YoloPool::YoloPool(const std::string& modelname, size_t count): modelName(modelname){
    YoloFactory::getInstance().setName(modelname);
    // 构造函数内容（如果有的话）
    for (size_t i = 0; i < count; i++) {
        pool.push(YoloFactory::getInstance().createYolo());
    }
}

YoloPool::~YoloPool() {
    // 析构函数内容（如果有的话）
    // for (size_t i = 0; i < pool.size(); i++) {
    //     pool[i]->stop();
    // }
}

std::shared_ptr<IYolo> YoloPool::acquire() {
    std::lock_guard<std::mutex> lock(poolMutex);
    if (pool.empty()) {
        LOG_INFO("yolo pool acquire new");
        // return YoloFactory::createYolo(YoloType::yolov5);
        // return YoloFactory::createYolo();
        return YoloFactory::getInstance().createYolo();
        // return std::make_unique<Yolov5>(modelName); // 创建新实例

    } else {
        LOG_INFO("yolo pool acquire");
        auto yolo = pool.front();
        pool.pop();
        return yolo;
    }
}

void YoloPool::release(std::shared_ptr<IYolo> yolo) {
    std::lock_guard<std::mutex> lock(poolMutex);
    pool.push(yolo);
    LOG_INFO("yolo pool release\n");
}
