#ifndef __PREPROCESS_GPU_H_
#define __PREPROCESS_GPU_H_
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
void preprocess_gpu(const unsigned char* d_input, float* d_output, const float* d_matrix, int original_width, int original_height, int target_width, int target_height, const float* d_mean, const float* d_std);
int preprocess_gpu(const unsigned char* d_input, float* d_output, int original_width, int original_height, int target_width, int target_height, const float* d_matrix, const float* d_mean, const float* d_std);
#endif