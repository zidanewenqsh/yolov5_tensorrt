#ifndef __YOLOPOOL__
#define __YOLOPOOL__
#ifndef YOLO_POOL_H
#define YOLO_POOL_H

#include <queue>
#include <mutex>
#include <memory>
#include "Yolo.h" // 假设 Yolo 类的定义在这个文件中

class YoloPool {
public:
    YoloPool(std::string modelname);
    std::unique_ptr<Yolov5> acquire();
    void release(std::unique_ptr<Yolov5> yolo);

private:
    std::queue<std::unique_ptr<Yolov5>> pool;
    std::mutex poolMutex;
    std::string modelName;
};

#endif // YOLO_POOL_H

#endif //__YOLOPOOL__