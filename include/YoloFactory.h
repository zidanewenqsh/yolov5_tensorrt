#ifndef __YOLOFACTORY_H__
#define __YOLOFACTORY_H__
#include "TensorrtModel.h"
#include "Yolov5.h"
#include <iostream>
#include <memory>
#include <string>

typedef TensorRTModel IYolo;
enum class YoloType {
    yolov5,
    // YoloV4
    // 其他类型...
};

// class YoloFactory {
// public:
//     static std::shared_ptr<IYolo> createYolo();
// };

// YoloFactory.h


// #include "IYolo.h"
// #include <memory>
// #include <string>

class YoloFactory {
public:
    static YoloFactory& getInstance();

    void setName(const std::string& newName);

    std::shared_ptr<IYolo> createYolo();

private:
    std::string name;
    YoloFactory() {}
};













#endif //__YOLOFACTORY_H__
