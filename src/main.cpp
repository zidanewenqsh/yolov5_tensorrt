#include "InferenceManager.h"
#include "Yolo.h"
#include "YoloUtils.h"
#include <opencv2/opencv.hpp>
#include "MemoryPool.h"
#include "MemoryPoolGpu.h"
#include "Logger.h"
// std::string extractFilename(const std::string& filePath) {
//     // 查找最后一个路径分隔符
//     size_t pos = filePath.find_last_of("/\\");
//     if (pos != std::string::npos) {
//         return filePath.substr(pos + 1);
//     }
//     return filePath; // 如果没有找到分隔符，返回整个字符串
// }
int main() {
#if 0
    std::string name = kModelName;
    Yolov5 yolo(name, 1<<28);
    std::string img_file = "bus.jpg";
    // yolo.forward_image(img_file);
    // printf("totalsize:%d\n", totalsize);
    std::string txtfile = "output.txt";
    auto imgfiles = readimgfilesFromFile(txtfile);
    for (auto imgfile:imgfiles) {
        printf("imgpath:%s\n", imgfile.c_str());
        yolo.forward_image(imgfile);
    }
#endif
#if 0
    size_t size = 1 << 25;
    MemPool cpumem(size);
    MemPoolGpu gpumem(size);
    void *cpuptr = cpumem.allocate();
    void *gpuptr = gpumem.allocate();


    printf("cpuptr:%p\n", cpuptr);
    printf("gpuptr:%p\n", gpuptr);
    std::string name = "yolov5s";
    InferenceManager infermagager(1<<24, 4, name);
    // auto yolo = infermagager.yoloPool.acquire();
    Yolov5 yolo(name);
    yolo.setMemory(cpuptr, gpuptr, size);
    // return 0;
    // yolo.forward_image(img_file);
    // printf("totalsize:%d\n", totalsize);
    std::string txtfile = "output.txt";
    auto imgfiles = readLinesFromFile(txtfile);
    for (auto imgfile:imgfiles) {
        printf("imgpath:%s\n", imgfile.c_str());
    
        auto imgname = extractFilename(imgfile);
        printf("imgname:%s\n", imgname.c_str());
        std::string savepath = "result_" + imgname; 
        printf("savepath:%s\n", savepath.c_str());
        // 处理一张图片
        // cv::Mat img = cv::imread(img_file);
        cv::Mat img = cv::imread(imgfile);
        if (img.empty()) {
            printf("cv::imread %s failed\n", imgfile.c_str());
            return -1;
        }
        // std::shared_ptr<Data> data = std::make_shared<ImageData>
        std::cout << "img size:" << img.size() << std::endl;
        std::cout << "img total:" << img.total() << std::endl;
        std::cout << "img channels:" << img.channels() << std::endl; 
        std::cout << "img elemSize:" << img.elemSize() << std::endl;
        

        std::shared_ptr<Data> data = std::make_shared<ImageData_t>();
        ImageData_t* rawPtr = dynamic_cast<ImageData_t*>(data.get());
        yolo.make_imagedata(img, rawPtr);
        std::cout << "img numel:" << rawPtr->numel << std::endl;
        std::cout << "img size:" << rawPtr->size << std::endl;
        std::cout << "img size2:" << img.total() * img.channels() << std::endl;
        std::cout << "rawPtr->width:" << rawPtr->width << std::endl;        
        yolo.inference(data.get());
        yolo.drawimg(img, savepath);
        // auto imgname = extractFilename(imgfile);
        // printf("imgname:%s\n", imgname.c_str());
        // std::string savepath = "result_" + imgname; 
        // printf("savepath:%s\n", savepath.c_str());
        // yolo.forward_image(imgfile);
    }
    #else
// #endif 
// #if 1
    Logger logger;
    logger.SetLevel(LogLevel::INFO);
    std::string name = "yolov5s";
    InferenceManager infermagager(1<<25, 8, name);
    std::string txtfile = "coco128_1.txt";
    auto imgfiles = readLinesFromFile(txtfile);
    std::vector<Image> images;
    for (auto imgfile:imgfiles) {
        // printf("imgpath:%s\n", imgfile.c_str());
        LOGINFO("imgpath:%s\n", imgfile.c_str());
    
        auto imgname = extractFilename(imgfile);
        // printf("imgname:%s\n", imgname.c_str());
        std::string savepath = "./output/result_" + imgname; 
        // printf("savepath:%s\n", savepath.c_str());
        
        // 处理一张图片
        cv::Mat img = cv::imread(imgfile);
        if (img.empty()) {
            // printf("cv::imread %s failed\n", imgfile.c_str());
            LOGERROR("cv::imread %s failed\n", imgfile.c_str());
            continue;
            // return -1;
        }
        Image image;
        image.img = img;
        image.imgname = imgname;
        image.srcpath = imgfile;
        image.savepath = savepath;
        std::cout << "img size:" << img.size() << std::endl;
        images.push_back(image);
        // infermagager.processSingleImage(img);
    }
    auto start = std::chrono::high_resolution_clock::now();
    infermagager.processImages(images);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Total Elapsed time: " << elapsed.count() << " seconds" << std::endl;
#endif
    return 0;
}