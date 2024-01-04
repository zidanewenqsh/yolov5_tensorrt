#include "postprocess.h"

// coco数据集的labels，关于coco：https://cocodataset.org/#home
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

static float iou(const Box& a, const Box& b){
    float cross_x1 = std::max(a.x1, b.x1);
    float cross_y1 = std::max(a.y1, b.y1);
    float cross_x2 = std::min(a.x2, b.x2);
    float cross_y2 = std::min(a.y2, b.y2);

    float cross_area = std::max(0.0f, cross_x2 - cross_x1) * std::max(0.0f, cross_y2 - cross_y1);
    float union_area = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1) 
                     + std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1) - cross_area;
    if(cross_area == 0 || union_area == 0) return 0.0f;
    return cross_area / union_area;
};

static std::vector<Box> nms(std::vector<Box> &bboxes, const float iou_threshold) {
    std::sort(bboxes.begin(), bboxes.end(), [](const Box &a, const Box &b) {
        return a.prob > b.prob;
    });
    std::vector<Box> res;
    std::vector<bool> flag(bboxes.size(), false);
    for (unsigned int i = 0; i < bboxes.size(); i++) {
        if (flag[i]) {
            continue;
        }
        res.push_back(bboxes[i]);
        for (unsigned int j = i + 1; j < bboxes.size(); j++) {
            if (flag[j] || bboxes[i].label != bboxes[j].label) {
                continue;
            }
            if (iou(bboxes[i], bboxes[j]) > iou_threshold) {
                flag[j] = true;
            }
        }
    }
    return res;
}

std::vector<Box> postprocess_cpu(float *data, int output_batch, int output_numbox, int output_numprob, 
        float confidence_threshold, float nms_threshold) { 
    std::vector<Box> bboxes;
    int num_classes = output_numprob - 5;
    // int output_numel = output_batch * output_numbox * output_numprob;
    // printf("output_numel: %d\n", output_numel);
    for (int i = 0; i < output_batch; i++) {
        float *pBatch = data + i * output_numbox * output_numprob;
        for (int j = 0; j < output_numbox; j++) {
            float *pBox = pBatch + j * output_numprob;
            // float *box_end = box_start + output_numprob;
            float prob = pBox[4];
            // printf("prob: %f\n", prob);
            // break;
            if (prob < confidence_threshold) {
                continue;
            }
            float *pClasses = pBox + 5;
            int label = std::max_element(pClasses, pClasses + num_classes) - pClasses;
            prob *= pClasses[label];
            if (prob < confidence_threshold) {
                continue;
            }
            float x1 = pBox[0] - pBox[2] / 2;
            float y1 = pBox[1] - pBox[3] / 2;
            float x2 = pBox[0] + pBox[2] / 2;
            float y2 = pBox[1] + pBox[3] / 2;
            Box box = {x1, y1, x2, y2, prob, label};
            bboxes.push_back(box);
        }
    }
    std::vector<Box> res = nms(bboxes, nms_threshold);
    return res;
}
cv::Mat draw_cpu(std::vector<Box> &boxes, cv::Mat &img, float *d2i) {
    cv::Mat img_draw = img.clone();
    for (auto box:boxes) {
        int x1 = d2i[0] * box.x1 + d2i[1] * box.y1 + d2i[2];
        int y1 = d2i[3] * box.x1 + d2i[4] * box.y1 + d2i[5];
        int x2 = d2i[0] * box.x2 + d2i[1] * box.y2 + d2i[2];
        int y2 = d2i[3] * box.x2 + d2i[4] * box.y2 + d2i[5];
        auto name  = cocolabels[box.label];
        auto caption   = cv::format("%s %.2f", name, box.prob);
        auto color = colors[box.label % 10];
        cv::rectangle(img_draw, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);
        // cv::putText(img_draw, std::to_string(box.label), cv::Point(x1, y1), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        cv::putText(img_draw, caption, cv::Point(x1, y1), cv::FONT_HERSHEY_SIMPLEX, 1, color, 2);
    }
    return img_draw;
}