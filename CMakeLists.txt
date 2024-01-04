cmake_minimum_required(VERSION 3.12)
project(yolov5_tensorrt)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)

# 设置 CUDA 支持
enable_language(CUDA)
# 设置默认的 CUDA 架构
set(CMAKE_CUDA_ARCHITECTURES 61) # 以您的 GPU 架构为准，例如 75 代表 Tesla T4

# # 设置输出目录
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
# 调试标志
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()


# 设置编译器选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_61") # 根据您的 GPU 架构调整
# 查找 Boost
# find_package(Boost REQUIRED COMPONENTS timer)
# 查找 OpenCV
find_package(OpenCV REQUIRED)

# 查找 CUDA
find_package(CUDA REQUIRED)
# 查找 TensorRT
# 注意：您可能需要根据 TensorRT 的安装情况调整此部分
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
          HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
          PATH_SUFFIXES include)
find_library(TENSORRT_LIBRARY nvinfer
             HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
             PATH_SUFFIXES lib lib64 lib/x64)

message(STATUS "\${CUDA_TOOLKIT_ROOT_DIR}=${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "\${CUDA_INCLUDE_DIRS}=${CUDA_INCLUDE_DIRS}")
message(STATUS "\${TENSORRT_INCLUDE_DIR}=${TENSORRT_INCLUDE_DIR}")
message(STATUS "\${OpenCV_INCLUDE_DIRS}=${OpenCV_INCLUDE_DIRS}")
message(STATUS "\${Boost_INCLUDE_DIR}=${Boost_INCLUDE_DIR}")
message(STATUS "\${CUDA_LIBRARIES}=${CUDA_LIBRARIES}")
message(STATUS "\${OpenCV_LIBS}=${OpenCV_LIBS}")
message(STATUS "\${TENSORRT_LIBRARY}=${TENSORRT_LIBRARY}")
message(STATUS "\${Boost_LIBRARIES}=${Boost_LIBRARIES}")

# 指定 include 目录
# include_directories(${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${Boost_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR})
include_directories(include include/common)
# 指定库文件目录
link_directories(lib /usr/local/lib /usr/lib/x86_64-linux-gnu/ /usr/local/cuda/lib64 /usr/local/lib64)
# link_directories(lib /usr/local/lib /usr/lib/x86_64-linux-gnu/ /usr/local/cuda/lib64 /usr/local/lib64)

# 设置环境变量 LD_LIBRARY_PATH
set(ENV{LD_LIBRARY_PATH} "/usr/local/lib:$ENV{LD_LIBRARY_PATH}")
# 自动收集 src 目录下的所有 .cpp 和 .cu 文件
file(GLOB_RECURSE SOURCE_FILES 
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")

# 添加可执行文件
# add_executable(MyExecutable ${SOURCE_FILES})
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
# target_link_libraries(${EXE_NAME} opencv_core opencv_imgproc opencv_imgcodecs m nvonnxparser)
target_link_libraries(${PROJECT_NAME} m nvonnxparser)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${TENSORRT_LIBRARY} ${Boost_LIBRARIES})

# # 指定源文件目录
# # file(GLOB_RECURSE SRC_FILES "src/*.c" "src/*.cpp" "src/*.cu")
# file(GLOB SRC_FILES "src/*.c" "src/*.cpp" "src/*.cu" "test/*.c" "test/*.cpp" "test/*.cu")

# # 为每个源文件创建一个可执行目标
# foreach(SOURCE_FILE ${SRC_FILES})
#     get_filename_component(EXE_NAME ${SOURCE_FILE} NAME_WE)
#     add_executable(${EXE_NAME} ${SOURCE_FILE})
#     # target_link_libraries(${EXE_NAME} opencv_core opencv_imgproc opencv_imgcodecs m nvonnxparser)
#     target_link_libraries(${EXE_NAME} m nvonnxparser)
#     target_link_libraries(${EXE_NAME} ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${TENSORRT_LIBRARY} ${Boost_LIBRARIES})
#     # target_link_libraries(${EXE_NAME} ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${TENSORRT_LIBRARY})
# endforeach()

# # 遍历 src 目录下的一级子目录
# file(GLOB CHILD_DIRS "src/*")
# foreach(DIR ${CHILD_DIRS})
#     if(IS_DIRECTORY ${DIR})
#         # 获取目录名称作为可执行文件的名称
#         get_filename_component(DIR_NAME ${DIR} NAME)

#         # 收集该目录下的所有源文件
#         file(GLOB DIR_SRC_FILES "${DIR}/*.c" "${DIR}/*.cpp" "${DIR}/*.cu")

#         # 为这些源文件创建一个可执行目标
#         add_executable(${DIR_NAME} ${DIR_SRC_FILES})
#         target_link_libraries(${DIR_NAME} ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${TENSORRT_LIBRARY} m nvonnxparser)
#         target_link_libraries(${DIR_NAME} ${CUDA_LIBRARIES} ${TENSORRT_LIBRARY} m nvonnxparser opencv_core opencv_imgproc opencv_imgcodecs)
#         # ... 其他链接库 ...
#     endif()
# endforeach()