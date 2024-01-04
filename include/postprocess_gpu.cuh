#ifndef __POSTPROCESS_GPU__
#define __POSTPROCESS_GPU__
#include <atomic>
#include <cuda_runtime.h>
#include <memory>
#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>

typedef struct gbox_s {
    float x1, y1, x2, y2;
    float prob; // 概率
    int label; // 类别
    int remove; // 是否保留
    void print() {
        printf("x1: %f, y1: %f, x2: %f, y2: %f, prob: %f, label: %d, remove: %d\n", x1, y1, x2, y2, prob, label, remove);
    }
} gBox;

void postprocess_cuda(float* d_data, gBox *d_filtered_boxes, int *d_box_count, float *d_matrix, 
    int output_numbox, int output_numprob, float confidence_threshold, float nms_threshold);
int postprocess_cuda(float* d_data, gBox *d_filtered_boxes, int *d_box_count, int output_numbox, int output_numprob, 
    float confidence_threshold, float nms_threshold, float *d_matrix);
int postprocess_cuda(float* d_data, gBox *d_filtered_boxes, int *d_box_count, int *h_box_count, int output_batch, int output_numbox, int output_numprob, 
    float confidence_threshold, float nms_threshold, float *d_matrix);
int postprocess_cuda(float* d_data, char *d_filtered_boxes, float *d_matrix, int output_batch, int output_numbox, int output_numprob, 
    float confidence_threshold, float nms_threshold, bool cpunms=false);
cv::Mat draw_g(gBox *boxes, int count, cv::Mat img);
cv::Mat draw_gpu(gBox *boxes, int count, cv::Mat img);
cv::Mat draw_gpu(char* h_filtered_boxes, cv::Mat img);
void nms_cpu(char *h_filtered_boxes, float nms_threshold);
#endif