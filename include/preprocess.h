#ifndef __PREPROCESS_H__
#define __PREPROCESS_H__
#include <opencv2/opencv.hpp>
#include <vector>
std::unique_ptr<float[]> calculate_matrix(int width, int height, int w, int h);
std::unique_ptr<float[]> calculate_invmatrix(int width, int height, int w, int h);
void preprocess_opencv_cpu(const cv::Mat& img, float *ret, float *matrix, int w, int h, bool norm=false);
void preprocess_opencv_cpu(const cv::Mat& img, float *ret, float* i2d, int w, int h, float *mean, float *std);
void preprocess_cpu_v2(const cv::Mat& img, float *ret, float* matrix, int w, int h, bool norm=false);
#endif