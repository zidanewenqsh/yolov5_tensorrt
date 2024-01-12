#include "YoloUtils.h"

void insertImageInfo(const std::shared_ptr<sql::Connection>& conn, const std::string& file_name, const std::string& upload_time, const std::string& source) {
    // std::shared_ptr<sql::PreparedStatement> pstmt;
    // 使用 std::shared_ptr 的构造函数来创建 pstmt
    // pstmt = conn->prepareStatement("INSERT INTO images (file_name, upload_time, source) VALUES (?, ?, ?)");
    std::shared_ptr<sql::PreparedStatement> pstmt(conn->prepareStatement("INSERT INTO images (file_name, upload_time, source) VALUES (?, ?, ?)"), [](sql::PreparedStatement* ptr){ delete ptr; });
    pstmt->setString(1, file_name);
    pstmt->setString(2, upload_time);
    pstmt->setString(3, source);
    pstmt->executeUpdate();
}
void insertDetectionResult(const std::shared_ptr<sql::Connection>& conn, int image_id, const std::string& class_name, double confidence, int bbox_xmin, int bbox_ymin, int bbox_xmax, int bbox_ymax, const std::string& detection_time) {
    // std::shared_ptr<sql::PreparedStatement> pstmt;
    // pstmt = conn->prepareStatement("INSERT INTO detections (image_id, class_name, confidence, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, detection_time) VALUES (?, ?, ?, ?, ?, ?, ?, ?)");
    // 使用 std::shared_ptr 的构造函数来创建 pstmt
    std::shared_ptr<sql::PreparedStatement> pstmt(conn->prepareStatement("INSERT INTO detections (image_id, class_name, confidence, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, detection_time) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"), [](sql::PreparedStatement* ptr){ delete ptr; });

    pstmt->setInt(1, image_id);
    pstmt->setString(2, class_name);
    pstmt->setDouble(3, confidence);
    pstmt->setInt(4, bbox_xmin);
    pstmt->setInt(5, bbox_ymin);
    pstmt->setInt(6, bbox_xmax);
    pstmt->setInt(7, bbox_ymax);
    pstmt->setString(8, detection_time);
    pstmt->executeUpdate();
}

std::string getCurrentDateTime() {
    // 获取当前时间点
    auto now = std::chrono::system_clock::now();
    // 转换为时间_t类型
    auto now_c = std::chrono::system_clock::to_time_t(now);
    // 转换为tm结构
    std::tm now_tm = *std::localtime(&now_c);

    // 使用stringstream进行格式化
    std::stringstream ss;
    ss << std::put_time(&now_tm, "%Y-%m-%d %H:%M:%S");
    return ss.str();
}
std::vector<std::string> readLinesFromFile(const std::string& filename) {
    std::vector<std::string> lines;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return lines; // Return an empty vector if file can't be opened
    }

    std::string line;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }

    file.close();
    return lines;
}
void saveWeight(const std::string &filename, const float *data, const int size) {
    std::ofstream outfile(filename, std::ios::binary);
    assert(outfile.is_open() && "saveData failed");
    outfile.write(reinterpret_cast<const char *>(&size), sizeof(int));
    outfile.write(reinterpret_cast<const char *>(data), size * sizeof(float));
    outfile.close();
}

std::vector<float> loadWeight(const std::string &filename) {
    std::ifstream infile(filename, std::ios::binary);
    assert(infile.is_open() && "loadWeight failed");
    int size;
    infile.read(reinterpret_cast<char *>(&size), sizeof(int));
    std::vector<float> data(size);
    infile.read(reinterpret_cast<char *>(data.data()), size * sizeof(float));
    infile.close();
    return data;
}

void saveData(const std::string &filename, const char *data, const int size) {
    std::ofstream outfile(filename, std::ios::binary);
    assert(outfile.is_open() && "saveData failed");
    // print(size);
    outfile.write(reinterpret_cast<const char *>(data), size);
    outfile.close();
}

std::vector<char> loadData(const std::string &filename) {
    std::ifstream infile(filename, std::ios::binary);
    // assert(infile.is_open() && "loadData failed");
    if (!infile.is_open()) {
        throw std::runtime_error("loadData failed: " + filename);
        // 或者使用 std::cerr 来打印错误消息，然后返回空向量或执行其他错误处理
        // std::cerr << "loadData failed: " + filename << std::endl;
        // return std::vector<char>();
    }
    infile.seekg(0, std::ios::end);
    int size = infile.tellg();
    infile.seekg(0, std::ios::beg);
    std::vector<char> data(size);
    infile.read(reinterpret_cast<char *>(data.data()), size);
    infile.close();
    return data;
}

void printDims(const nvinfer1::Dims& dims) {
    std::cout << "Dimensions: ";
    for (int i = 0; i < dims.nbDims; ++i) {
        std::cout << dims.d[i];
        if (i < dims.nbDims - 1) {
            std::cout << " x ";
        }
    }
    std::cout << std::endl;
}
void calculate_matrix(float* matrix, int width, int height, int w, int h) {
    float scale = std::min((float)w/width, (float)h/height);
    // float *matrix = new float[12];
    float *i2d = matrix;
    float *d2i = matrix + 6;
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
}

std::string extractFilename(const std::string& filePath) {
    // 查找最后一个路径分隔符
    size_t pos = filePath.find_last_of("/\\");
    if (pos != std::string::npos) {
        return filePath.substr(pos + 1);
    }
    return filePath; // 如果没有找到分隔符，返回整个字符串
}
// 函数：从文件的绝对路径中提取没有扩展名的文件名
std::string extractFilenameWithoutExtension(const std::string& filePath) {
    // 查找最后一个路径分隔符
    size_t lastSlashPos = filePath.find_last_of("/\\");
    std::string filename = (lastSlashPos != std::string::npos) ? filePath.substr(lastSlashPos + 1) : filePath;

    // 查找最后一个点，它可能是文件扩展名的开始
    size_t lastDotPos = filename.find_last_of('.');
    if (lastDotPos != std::string::npos && lastDotPos > 0) {
        // 返回没有扩展名的文件名部分
        return filename.substr(0, lastDotPos);
    }
    return filename; // 如果没有找到点，返回整个文件名
}

// c++ 17
// std::string extractFilename(const std::string& filePath) {
//     std::filesystem::path p(filePath);
//     return p.filename().string();
// }
