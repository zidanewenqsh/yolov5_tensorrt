#ifndef __YOLOPOOL__
#define __YOLOPOOL__
#include "Logger.h"
#ifndef YOLO_POOL_H
#define YOLO_POOL_H

#include <queue>
#include <mutex>
#include <memory>
// #include "IYolo.h" // 假设 Yolo 类的定义在这个文件中
#include "TensorrtModel.h" // 假设 Yolo 类的定义在这个文件中

typedef TensorRTModel IYolo;
class YoloPool:public Logger {
public:
    YoloPool(const std::string& modelname, size_t count); // 构造函数, count 为预先创建的 Yolo 实例的数量
    ~YoloPool();
    std::shared_ptr<IYolo> acquire();
    void release(std::shared_ptr<IYolo> yolo);

private:
    std::queue<std::shared_ptr<IYolo>> pool;
    std::mutex poolMutex;
    std::string modelName;
};

#endif // YOLO_POOL_H

#endif //__YOLOPOOL__