#include "InferenceManager.h"
// #include "Yolo.h"
#include "YoloUtils.h"
#include <opencv2/opencv.hpp>
#include "MemoryPool.h"
#include "MemoryPoolGpu.h"
#include "Logger.h"
int main(int argc, char* argv[]) {
    Logger logger;
    logger.SetLevel(LogLevel::DEBUG);
    // std::string name = "yolov5s_batch2";
    std::string name = "yolov5s";
    // std::string name = "yolov5s";
    if (argc > 1) {
        name = argv[1];
    }
    InferenceManager infermagager(1<<28, 1, name);
    std::string txtfile = "coco128_1.txt";
    // std::string txtfile = "input.txt";
    auto imgfiles = readLinesFromFile(txtfile);
    std::vector<Image> images;
    for (auto imgfile:imgfiles) {
        // printf("imgpath:%s\n", imgfile.c_str());
        // LOGDEBUG("imgpath:%s\n", imgfile.c_str());
    
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
    std::cout << images.size() << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    #if 0
    infermagager.processImages(images);
    #else
    // infermagager.processBatchImage(images);
    infermagager.processBatch(images, 5);
    #endif
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Total Elapsed time: " << elapsed.count() << " seconds" << std::endl;
    return 0;
}