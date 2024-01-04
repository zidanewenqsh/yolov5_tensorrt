#ifndef __POSTPROCESS_H__
#define __POSTPROCESS_H__
#include <cstdio>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <memory>

typedef struct box_s {
    float x1, y1, x2, y2;
    float prob; // 概率
    int label; // 类别
    void print() {
        printf("x1: %f, y1: %f, x2: %f, y2: %f, prob: %f, label: %d\n", x1, y1, x2, y2, prob, label);
    }
}Box;

std::vector<Box> postprocess_cpu(float *data, int output_batch, int output_numbox, int output_numprob, 
        float confidence_threshold, float nms_threshold);
cv::Mat draw(std::vector<Box> &boxes, cv::Mat &img, float *d2i);       
cv::Mat draw_cpu(std::vector<Box> &boxes, cv::Mat &img, float *d2i);       

#endif