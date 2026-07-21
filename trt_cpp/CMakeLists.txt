cmake_minimum_required(VERSION 3.18)
project(trt_detect CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# TensorRT: 头文件来自 OSS 仓库, 库来自 venv pip 包 (tensorrt_cu12_libs)
set(TRT_INCLUDE ${CMAKE_SOURCE_DIR}/third_party/trt_oss/include)
set(TRT_LIB ${CMAKE_SOURCE_DIR}/../venv/lib/python3.12/site-packages/tensorrt_libs)

# CUDA
set(CUDA_ROOT /usr/local/cuda)

find_package(OpenCV REQUIRED)

add_executable(trt_detect main.cpp)
target_include_directories(trt_detect PRIVATE
    ${TRT_INCLUDE}
    ${CUDA_ROOT}/include
    ${OpenCV_INCLUDE_DIRS})
target_link_directories(trt_detect PRIVATE ${TRT_LIB} ${CUDA_ROOT}/lib64)
# pip 包内只有 .so.11 没有 .so 符号链接, 用 -l: 直接指定文件名
target_link_libraries(trt_detect
    -l:libnvinfer.so.11
    -l:libnvonnxparser.so.11
    cudart
    ${OpenCV_LIBS})
# 运行时能找到 venv 里的 TensorRT 库
set_target_properties(trt_detect PROPERTIES
    BUILD_RPATH "${TRT_LIB};${CUDA_ROOT}/lib64"
    INSTALL_RPATH "${TRT_LIB};${CUDA_ROOT}/lib64")
