#include "postprocess_gpu.h"
#define INNERNMS 0
// #define GPUNMS 0
#define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)
static bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line){
    if(code != cudaSuccess){    
        const char* err_name = cudaGetErrorName(code);    
        const char* err_message = cudaGetErrorString(code);  
        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);   
        return false;
    }
    return true;
}
static const char* cocolabels[] = {
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
};

static std::vector<cv::Scalar> colors = {
    cv::Scalar(255, 0, 0),      // 蓝色
    cv::Scalar(0, 255, 0),      // 绿色
    cv::Scalar(0, 0, 255),      // 红色
    cv::Scalar(0, 255, 255),    // 黄色
    cv::Scalar(255, 0, 255),    // 洋红色（品红）
    cv::Scalar(255, 255, 0),    // 青色
    cv::Scalar(0, 165, 255),    // 橙色
    cv::Scalar(128, 0, 128),    // 紫色
    cv::Scalar(255, 192, 203),  // 粉色
    cv::Scalar(128, 128, 128)   // 灰色
};

__global__ void filter_boxes_kernel_withnms(float* boxes, char *d_filtered_boxes, float* d_matrix, int num_boxes, int num_probs, 
    float conf_threshold, float nms_threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_boxes) return ;

    float* box = boxes + idx * num_probs;
    float prob = box[4]; // 置信度

    if (prob < conf_threshold) return;

    // 找到最大类别概率
    float max_class_prob = 0.0f;
    int label = -1;
    for (int i = 5; i < num_probs; ++i) {
        if (box[i] > max_class_prob) {
            max_class_prob = box[i];
            label = i - 5;
        }
    }

    prob *= max_class_prob;
    if (prob < conf_threshold) return;

    // int count = atomicAdd(box_count, 1);
    // gBox *bboxes = filtered_boxes + count;
    int count = atomicAdd((int *)d_filtered_boxes, 1);
    gBox *gboxes = (gBox*)(d_filtered_boxes + sizeof(int));
    gBox *bboxes = gboxes + count;
    // 计算边界框坐标
    float x1 = box[0] - box[2] / 2;
    float y1 = box[1] - box[3] / 2;
    float x2 = box[0] + box[2] / 2;
    float y2 = box[1] + box[3] / 2;
#if 1
    x1 = d_matrix[0] * x1 + d_matrix[1] * y1 + d_matrix[2];
    y1 = d_matrix[3] * x1 + d_matrix[4] * y1 + d_matrix[5];
    x2 = d_matrix[0] * x2 + d_matrix[1] * y2 + d_matrix[2];
    y2 = d_matrix[3] * x2 + d_matrix[4] * y2 + d_matrix[5];
#endif
    bboxes->x1 = x1;
    bboxes->y1 = y1;
    bboxes->x2 = x2;
    bboxes->y2 = y2;
    bboxes->prob = prob;
    bboxes->label = label;
    // printf("x1: %f, y1: %f, x2: %f, y2: %f, prob: %f, label: %d, count: %d\n", x1, y1, x2, y2, prob, label, count);
    __syncthreads();
    for (int i = 0; i < count - 1; i++) {
        // gBox *bbox = d_filtered_boxes + i;
        gBox *bbox = gboxes + i;
        if (bboxes->remove || bbox->remove || bbox->label != label) continue;
        float xx1 = max(bbox->x1, x1);
        float yy1 = max(bbox->y1, y1);
        float xx2 = min(bbox->x2, x2);
        float yy2 = min(bbox->y2, y2);
        float w = max(0.0f, xx2 - xx1 + 1);
        float h = max(0.0f, yy2 - yy1 + 1);
        float inter = w * h;
        if (inter == 0) continue;
        float ovr = inter / ((bbox->x2 - bbox->x1 + 1) * (bbox->y2 - bbox->y1 + 1) + (x2 - x1 + 1) * (y2 - y1 + 1) - inter);
        if (ovr < nms_threshold) continue;
        if (prob > bbox->prob) {
            bbox->remove = 1;
        } else {
            bboxes->remove = 1;
        }
    }
}

__global__ void filter_boxes_kernel(float* boxes, char *d_filtered_boxes, float* d_matrix, int num_boxes, int num_probs, 
    float conf_threshold, float nms_threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_boxes) return ;

    float* box = boxes + idx * num_probs;
    float prob = box[4]; // 置信度

    if (prob < conf_threshold) return;

    // 找到最大类别概率
    float max_class_prob = 0.0f;
    int label = -1;
    for (int i = 5; i < num_probs; ++i) {
        if (box[i] > max_class_prob) {
            max_class_prob = box[i];
            label = i - 5;
        }
    }

    prob *= max_class_prob;
    if (prob < conf_threshold) return;

    int count = atomicAdd((int *)d_filtered_boxes, 1);
    gBox *bboxes = (gBox*)(d_filtered_boxes + sizeof(int)) + count;
    // 计算边界框坐标
    float x1 = box[0] - box[2] / 2;
    float y1 = box[1] - box[3] / 2;
    float x2 = box[0] + box[2] / 2;
    float y2 = box[1] + box[3] / 2;
    // printf("x1: %f, y1: %f, x2: %f, y2: %f, prob: %f, label: %d, count: %d\n", x1, y1, x2, y2, prob, label, count);
#if 1
    x1 = d_matrix[0] * x1 + d_matrix[1] * y1 + d_matrix[2];
    y1 = d_matrix[3] * x1 + d_matrix[4] * y1 + d_matrix[5];
    x2 = d_matrix[0] * x2 + d_matrix[1] * y2 + d_matrix[2];
    y2 = d_matrix[3] * x2 + d_matrix[4] * y2 + d_matrix[5];
#endif
    bboxes->x1 = x1;
    bboxes->y1 = y1;
    bboxes->x2 = x2;
    bboxes->y2 = y2;
    bboxes->prob = prob;
    bboxes->label = label;
    // printf("x1: %f, y1: %f, x2: %f, y2: %f, prob: %f, label: %d, count: %d\n", x1, y1, x2, y2, prob, label, count);
}

__global__ void nms(char *d_filtered_boxes, float nms_threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int box_count = *(int *)d_filtered_boxes;
    // printf("box_count:%d\n", box_count);
    if (idx >= box_count) return;
    gBox* box = (gBox*)(d_filtered_boxes + sizeof(int)) + idx;
    // printf("x1: %f, y1: %f, x2: %f, y2: %f, prob: %f, label: %d, count: %d\n", 
    //     box->x1, box->y1, box->x2, box->y2, box->prob, box->label, box_count);
    if (box->remove) return;
    for (int i = 0; i < box_count; i++) {
        if (i == idx) continue;
        gBox *bbox = (gBox*)(d_filtered_boxes + sizeof(int)) + i;
        if (bbox->remove || bbox->label != box->label) continue;
        float xx1 = max(bbox->x1, box->x1);
        float yy1 = max(bbox->y1, box->y1);
        float xx2 = min(bbox->x2, box->x2);
        float yy2 = min(bbox->y2, box->y2);
        float w = max(0.0f, xx2 - xx1 + 1);
        float h = max(0.0f, yy2 - yy1 + 1);
        float inter = w * h;
        if (inter == 0) continue;
        float ovr = inter / ((bbox->x2 - bbox->x1 + 1) * (bbox->y2 - bbox->y1 + 1) + 
                        (box->x2 - box->x1 + 1) * (box->y2 - box->y1 + 1) - inter);
        if (ovr < nms_threshold) continue;
        if (box->prob > bbox->prob) {
            bbox->remove = 1;
        } else {
            box->remove = 1;
        }
    }
}
void nms_cpu(char *h_filtered_boxes, float nms_threshold) {
    int box_count = *(int *)h_filtered_boxes;
    // printf("box_count:%d\n", box_count);
    // if (idx >= box_count) return;
    gBox* boxes = (gBox*)(h_filtered_boxes + sizeof(int));
    // printf("x1: %f, y1: %f, x2: %f, y2: %f, prob: %f, label: %d, count: %d\n", 
    //     box->x1, box->y1, box->x2, box->y2, box->prob, box->label, box_count);
    std::sort(boxes, boxes + box_count, [](const gBox& a, const gBox& b) {
        return a.prob > b.prob; // 降序排序
    });
    for (int i = 0; i < box_count; i++) {
        gBox *box = (gBox*)(h_filtered_boxes + sizeof(int)) + i;
        // printf("x1: %f, y1: %f, x2: %f, y2: %f, prob: %f, label: %d, remove: %d\n", 
        //     box->x1, box->y1, box->x2, box->y2, box->prob, box->label, box->remove);
        if (box->remove) continue;
        for (int j = i + 1; j < box_count; j++) {
            gBox *bbox = (gBox*)(h_filtered_boxes + sizeof(int)) + j;
            if (bbox->remove || box->label != bbox->label) continue;
            float xx1 = max(bbox->x1, box->x1);
            float yy1 = max(bbox->y1, box->y1);
            float xx2 = min(bbox->x2, box->x2);
            float yy2 = min(bbox->y2, box->y2);
            float w = max(0.0f, xx2 - xx1 + 1);
            float h = max(0.0f, yy2 - yy1 + 1);
            float inter = w * h;
            if (inter == 0) continue;
            float ovr = inter / ((bbox->x2 - bbox->x1 + 1) * (bbox->y2 - bbox->y1 + 1) + 
                            (box->x2 - box->x1 + 1) * (box->y2 - box->y1 + 1) - inter);
            if (ovr < nms_threshold) continue;
            bbox->remove = 1;
        }
    }
}
int postprocess_cuda(float* d_data, char *d_filtered_boxes, float *d_matrix, int output_batch, int output_numbox, int output_numprob, 
    float confidence_threshold, float nms_threshold, bool cpunms) {
    for (int i = 0; i < output_batch; i++) {
        // int output_numel = output_numbox * output_numprob;
        // 调用 Kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (output_numbox + threadsPerBlock - 1) / threadsPerBlock;
        cudaError_t error;
#if INNERNMS
        filter_boxes_kernel_withnms<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_filtered_boxes, d_matrix, 
            output_numbox, output_numprob, confidence_threshold, nms_threshold);
        // 检查是否有错误发生
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "Postprocess CUDA Kernel Error: " << cudaGetErrorString(error) << std::endl;
            return -1;
        }
        // cudaMemcpy(h_box_count, d_box_count, sizeof(int), cudaMemcpyDeviceToHost);
#else
        filter_boxes_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_filtered_boxes, d_matrix, 
            output_numbox, output_numprob, confidence_threshold, nms_threshold);
        // 检查是否有错误发生
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "Postprocess CUDA Kernel Error: " << cudaGetErrorString(error) << std::endl;
            return -1;
        }
        // 同步设备以确保所有操作都已完成
        error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            std::cerr << "Postprocess CUDA Error: " << cudaGetErrorString(error) << std::endl;
            return -1;
        }
        if (cpunms) return 0; 
        int h_box_count;
        cudaMemcpy(&h_box_count, d_filtered_boxes, sizeof(int), cudaMemcpyDeviceToHost);
        // cudaMemcpy(h_box_count+i, d_box_count+i, sizeof(int), cudaMemcpyDeviceToHost);
        // nms
        threadsPerBlock = 256;
        // blocksPerGrid = (*(h_box_count) + threadsPerBlock - 1) / threadsPerBlock;
        blocksPerGrid = (h_box_count + threadsPerBlock - 1) / threadsPerBlock;
        // nms<<<blocksPerGrid, threadsPerBlock>>>(d_filtered_boxes,  *h_box_count, nms_threshold);   
        nms<<<blocksPerGrid, threadsPerBlock>>>(d_filtered_boxes, nms_threshold);   

        // 检查是否有错误发生
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "Postprocess CUDA Kernel Error: " << cudaGetErrorString(error) << std::endl;
            return -1;
        }
#endif
    }
    return 0;
}


cv::Mat draw_gpu(char* h_filtered_boxes, cv::Mat img) {
    // cv::Mat img_draw = img.clone();
    int count = *(int *)h_filtered_boxes;
    gBox *boxes = (gBox*)(h_filtered_boxes + sizeof(int)); 
    for (int i = 0; i < count; i++) {
        auto box = boxes[i];
        if (box.remove == 1) continue;
        // printf("i: %d\n", i);
        // box.print();
        int x1 = (int)box.x1;
        int y1 = (int)box.y1;
        int x2 = (int)box.x2;
        int y2 = (int)box.y2;
        auto name = cocolabels[box.label];
        auto caption = cv::format("%s %.2f", name, box.prob);
        auto color = colors[box.label % 10];
        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);
        cv::putText(img, caption, cv::Point(x1, y1), cv::FONT_HERSHEY_SIMPLEX, 1, color, 2);
    }
    return img;
}