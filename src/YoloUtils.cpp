#include "YoloUtils.h"
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
    assert(infile.is_open() && "loadData failed");
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
