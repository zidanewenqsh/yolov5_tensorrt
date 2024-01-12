#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorrt as trt
print(trt.__version__)
def convert_onnx_to_tensorrt(onnx_file_path, tensorrt_file_path, width, height):
    """
    将ONNX模型转换为TensorRT模型。

    参数:
    onnx_file_path (str): ONNX模型文件的路径。
    tensorrt_file_path (str): 要保存的TensorRT模型文件的路径。

    返回:
    bool: 转换是否成功。
    """
    # 创建一个TensorRT引擎生成器和解析器
    # builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
    # network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    # parser = trt.OnnxParser(network, builder.logger)
    # builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
    builder = trt.Builder(trt.Logger(trt.Logger.INFO))
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, builder.logger)
    # 解析ONNX模型
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False

    # 配置生成器设置
    config = builder.create_builder_config()

    # 创建优化配置
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name

    min_shape = (1, 3, height, width)  # 最小形状
    optimal_shape = (5, 3, height, width)  # 最优形状
    max_shape = (10, 3, height, width)  # 最大形状

    profile.set_shape(input_name, min_shape, optimal_shape, max_shape)
    config.add_optimization_profile(profile)


    # config.max_workspace_size = 1 << 30  # 1GB的工作空间
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)  # 1GB的工作空间

    # 使用build_serialized_network生成序列化的TensorRT网络
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("Failed to build serialized engine.")
        return False

    # 保存序列化引擎到文件
    with open(tensorrt_file_path, "wb") as f:
        f.write(serialized_engine)

    return True
if __name__ == '__main__':
    # onnx_file_path = "resnet18.onnx"
    # tensorrt_file_path = "resnet18.trt"
    # convert_onnx_to_tensorrt(onnx_file_path, tensorrt_file_path, 224, 224)
    onnx_file_path = "build/yolov5s.onnx"
    tensorrt_file_path = "build/yolov5s.trt"
    convert_onnx_to_tensorrt(onnx_file_path, tensorrt_file_path, 640, 640)