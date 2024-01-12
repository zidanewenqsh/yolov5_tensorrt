#include "InferenceManager.h"
// #include "Yolo.h"
#include "YoloUtils.h"
#include <opencv2/opencv.hpp>
#include "MemoryPool.h"
#include "MemoryPoolGpu.h"
#include "Logger.h"
int main() {
    Logger logger;
    logger.SetLevel(LogLevel::INFO);
    std::string name = "yolov5s";
    InferenceManager infermagager(1<<25, 4, name);
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
        // std::cout << "img size:" << img.size() << std::endl;
        images.push_back(image);
        // infermagager.processSingleImage(img);
    }
    auto start = std::chrono::high_resolution_clock::now();
    infermagager.processImages(images);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Total Elapsed time: " << elapsed.count() << " seconds" << std::endl;
    return 0;
}