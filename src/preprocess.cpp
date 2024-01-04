#include <opencv2/opencv.hpp>
#include <vector>

// // letterbox函数的实现
// static cv::Mat letterbox(cv::Mat &src, int w, int h, float* i2d, float *d2i) {
//     int width = src.cols;
//     int height = src.rows;

//     float scale = std::min(float(w) / width, float(h) / height);
//     i2d[0] = scale;
//     i2d[1] = 0;
//     i2d[2] = (-width * scale + w) / 2;
//     i2d[3] = 0;
//     i2d[4] = scale;
//     i2d[5] = (-height * scale + h) / 2;

//     d2i[0] = 1 / scale;
//     d2i[1] = 0;
//     d2i[2] = (width * scale - w) / 2 / scale;
//     d2i[3] = 0;
//     d2i[4] = 1 / scale;
//     d2i[5] = (height * scale - h) / 2 / scale;

//     cv::Mat M(2, 3, CV_32F, i2d);
//     cv::Mat out = cv::Mat::zeros(h, w, CV_8UC3);
//     cv::warpAffine(src, out, M, out.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));  
//     return out;
// }

void preprocess_opencv_cpu(const cv::Mat& img, float *ret, float* i2d, int w, int h, float *mean, float* std) {
    // 创建图像副本
    cv::Mat img0 = img.clone();
    // 均值和标准差
    // 应用letterbox变换
    cv::Mat processed = cv::Mat::zeros(h, w, CV_8UC3);
    cv::Mat M(2, 3, CV_32F, i2d);
    cv::warpAffine(img0, processed, M, processed.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));  
    // processed = letterbox(processed, w, h, matrix, matrix + 6);
    // 如果图像是BGR格式，则转换为RGB，否则相反
    cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);

    // 转换为float32
    processed.convertTo(processed, CV_32F, 1.0 / 255.0);
    // processed.convertTo(processed, CV_32F, 1.0);
#if 0
    // 减去均值并除以标准差, hwc转为chw
    int channelSize = w * h;
    float *c1 = ret, *c2 = ret + channelSize, *c3 = ret + channelSize * 2;
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            *(c1++) = (processed.at<cv::Vec3f>(i, j)[0]-mean[0])/std[0];
            *(c2++) = (processed.at<cv::Vec3f>(i, j)[1]-mean[1])/std[1];
            *(c3++) = (processed.at<cv::Vec3f>(i, j)[2]-mean[2])/std[2];
        }
    }
#else
  // NHWC to NCHW：rgbrgbrgb to rrrgggbbb：
  std::vector<cv::Mat> channels;
  cv::split(processed, channels); // 将输入图像分解成三个单通道图像：rrrrr、ggggg、bbbbb

  // 将每个单通道图像进行reshape操作，变为1x1xHxW的四维矩阵
  for (int i = 0; i < 3; i++) {
    cv::Mat img = channels[i];
    // reshape参数分别是cn：通道数，rows：行数
    // 类似[[r,r,r,r,r]]或[[g,g,g,g,g]]或[[b,b,b,b,b]]，每个有width * height个元素
    channels[i] = (img.reshape(1, 1)-mean[i])/std[i];

  }
  // 将三个单通道图像拼接成一个三通道图像，即rrrrr、ggggg、bbbbb拼接成rrrgggbbb
  // cv::Mat warp_dst_nchw;
  cv::hconcat(channels, processed);
  // 将三通道图像转换为float32
//   ret = (float*)processed.data;
    memcpy(ret, processed.data, w * h * 3 * sizeof(float));
#endif
    return ;
}


void preprocess_opencv_cpu(const cv::Mat& img, float *ret, float* i2d, int w, int h, bool norm=false) {
    // 创建图像副本
    cv::Mat img0 = img.clone();
    // 均值和标准差
    float mean[3] = {0}, std[3] = {1, 1, 1};
    if (norm) {
        mean[0] = 0.485;
        mean[1] = 0.456;
        mean[2] = 0.406;
        std[0]  = 0.229;
        std[1]  = 0.224;
        std[2]  = 0.225;
    };

    // 应用letterbox变换
    cv::Mat processed = cv::Mat::zeros(h, w, CV_8UC3);
    cv::Mat M(2, 3, CV_32F, i2d);
    cv::warpAffine(img0, processed, M, processed.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));  
    // processed = letterbox(processed, w, h, matrix, matrix + 6);
    // 如果图像是BGR格式，则转换为RGB，否则相反
    cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);

    // 转换为float32
    processed.convertTo(processed, CV_32F, 1.0 / 255.0);
    // processed.convertTo(processed, CV_32F, 1.0);
#if 0
    // 减去均值并除以标准差, hwc转为chw
    int channelSize = w * h;
    float *c1 = ret, *c2 = ret + channelSize, *c3 = ret + channelSize * 2;
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            *(c1++) = (processed.at<cv::Vec3f>(i, j)[0]-mean[0])/std[0];
            *(c2++) = (processed.at<cv::Vec3f>(i, j)[1]-mean[1])/std[1];
            *(c3++) = (processed.at<cv::Vec3f>(i, j)[2]-mean[2])/std[2];
        }
    }
#else
  // NHWC to NCHW：rgbrgbrgb to rrrgggbbb：
  std::vector<cv::Mat> channels;
  cv::split(processed, channels); // 将输入图像分解成三个单通道图像：rrrrr、ggggg、bbbbb

  // 将每个单通道图像进行reshape操作，变为1x1xHxW的四维矩阵
  for (int i = 0; i < 3; i++) {
    cv::Mat img = channels[i];
    // reshape参数分别是cn：通道数，rows：行数
    // 类似[[r,r,r,r,r]]或[[g,g,g,g,g]]或[[b,b,b,b,b]]，每个有width * height个元素
    channels[i] = (img.reshape(1, 1)-mean[i])/std[i];

  }
  // 将三个单通道图像拼接成一个三通道图像，即rrrrr、ggggg、bbbbb拼接成rrrgggbbb
  // cv::Mat warp_dst_nchw;
  cv::hconcat(channels, processed);
  // 将三通道图像转换为float32
//   ret = (float*)processed.data;
    memcpy(ret, processed.data, w * h * 3 * sizeof(float));
#endif
    return ;
}

float bilinearInterpolateChannel(const cv::Mat& img, float x, float y, int channel, uchar borderValue = 114) {
    // printf("x: %f, y: %f\n", x, y);
    // 检查坐标是否超出图像边界
    if (x < 0 || y < 0 || x >= img.cols - 1 || y >= img.rows - 1) {
        return borderValue;  // 返回常数值
    }

    int x1 = static_cast<int>(x);
    int y1 = static_cast<int>(y);
    int x2 = x1 + 1;
    int y2 = y1 + 1;

    float a = x - x1;
    float b = y - y1;

    float inter1 = (1 - a) * img.at<cv::Vec3b>(y1, x1)[channel] + a * img.at<cv::Vec3b>(y1, x2)[channel];
    float inter2 = (1 - a) * img.at<cv::Vec3b>(y2, x1)[channel] + a * img.at<cv::Vec3b>(y2, x2)[channel];

    return (1 - b) * inter1 + b * inter2;
}


std::unique_ptr<float[]> calculate_matrix(int width, int height, int w, int h) {
    float scale = std::min((float)w/width, (float)h/height);
    // float *matrix = new float[12];
    auto matrix = std::unique_ptr<float[]>(new float[12]);
    float *i2d = matrix.get();
    float *d2i = matrix.get() + 6;
    i2d[0] = scale;
    i2d[1] = 0;
    i2d[2] = (-width * scale + w) / 2;
    i2d[3] = 0;
    i2d[4] = scale;
    i2d[5] = (-height * scale + h) / 2;
    d2i[0] = 1 / scale;
    d2i[1] = 0;
    d2i[2] = (width * scale - w) / 2 / scale;
    d2i[3] = 0 ;
    d2i[4] = 1 / scale;
    d2i[5] = (height * scale - h) / 2 / scale;
    return matrix;
}

std::unique_ptr<float[]> calculate_invmatrix(int width, int height, int w, int h) {
    float scale = std::min((float)w/width, (float)h/height);
    // float *matrix = new float[12];
    auto matrix = std::unique_ptr<float[]>(new float[6]);
    float *d2i = matrix.get();
    d2i[0] = 1 / scale;
    d2i[1] = 0;
    d2i[2] = (width * scale - w) / 2 / scale;
    d2i[3] = 0 ;
    d2i[4] = 1 / scale;
    d2i[5] = (height * scale - h) / 2 / scale;
    return matrix;
}

void preprocess_cpu_v2(const cv::Mat& img, float *ret, float* matrix, int w, int h, bool norm=false) {
    /* 双线性插值版本
    */
    // 创建图像副本
    cv::Mat processed = img.clone();
    // 均值和标准差
    float mean[3] = {0}, std[3] = {1, 1, 1};
    if (norm) {
        mean[0] = 0.485;
        mean[1] = 0.456;
        mean[2] = 0.406;
        std[0]  = 0.229;
        std[1]  = 0.224;
        std[2]  = 0.225;
    };
    float *d2i = matrix;

    int channelSize = w * h;

    float *c1 = ret, *c2 = ret + channelSize, *c3 = ret + channelSize * 2;
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            float x = d2i[0] * j + d2i[1] * i + d2i[2];
            float y = d2i[3] * j + d2i[4] * i + d2i[5];
            *(c3++) = (bilinearInterpolateChannel(img, x, y, 0, 114)/255-mean[0])/std[0];
            *(c2++) = (bilinearInterpolateChannel(img, x, y, 1, 114)/255-mean[1])/std[1];
            *(c1++) = (bilinearInterpolateChannel(img, x, y, 2, 114)/255-mean[2])/std[2];
        }
    }
    return ;
}
