#include "YoloFactory.h"
YoloFactory& YoloFactory::getInstance() {
    static YoloFactory instance;
    return instance;
}

void YoloFactory::setName(const std::string& newName) {
    name = newName;
}

std::shared_ptr<IYolo> YoloFactory::createYolo() {
    // 根据 name 创建相应的 Yolo 对象
    // 示例代码，根据实际情况调整
    return std::make_shared<Yolov5>(name);
    // if (name == "YoloV5") {
    //     return std::make_shared<YoloV5>();
    // }
    // 其他逻辑...
    return nullptr;
}
// std::shared_ptr<IYolo> YoloFactory::createYolo() {
//     return std::make_shared<Yolov5>();
    // switch (type) {
        // case YoloType::yolov5:
            // return std::make_shared<Yolov5>();
        // case YoloType::YoloV4:
        //     return std::make_shared<YoloV4>();
        // 其他类型...
    // }
    // return nullptr;
// }