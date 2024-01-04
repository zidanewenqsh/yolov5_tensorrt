#ifndef __YOLOCONFIG_H__
#define __YOLOCONFIG_H__

// const static char* kModelName = "yolov5s";
const static char* kInputTensorName = "images";
const static char* kOutputTensorName = "output0";
constexpr static int kInputH = 640;
constexpr static int kInputW = 640;
constexpr static int kImageHMax = 1024;
constexpr static int kImageWMax = 1024;
constexpr static int kChannel = 3;
constexpr static int kMaxBatch = 1;
const static float kNmsThresh = 0.5f;
const static float kConfThresh = 0.25f;


#endif // __YOLOCONFIG_H__