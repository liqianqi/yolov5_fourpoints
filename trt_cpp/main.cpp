// YOLOv5 四点装甲板检测 - TensorRT C++ 推理测试程序
//
// 功能:
//   1. 首次运行时从 FP16 ONNX 构建 engine 并缓存
//      (TensorRT 11 为强类型网络, 精度跟随 ONNX 数据类型, 故导出时用 --half)
//   2. 对目录内所有图片推理: letterbox 预处理 -> enqueueV3 -> 置信度过滤 + NMS
//   3. 绘制四边形与类别标签保存到输出目录, 统计延迟
//
// 模型输出: (1, 25200, 21) = 8 关键点像素坐标(640尺度) + 1 obj + 12 联合类别分数
//
// 用法: ./trt_detect <onnx_fp16> <engine> <images_dir> <out_dir> [conf=0.4] [iou=0.3]

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::high_resolution_clock;

static const int INPUT_W = 640, INPUT_H = 640;
static const int NUM_PRED = 25200, NUM_COLS = 21, NUM_CLS = 12;
static const char* CLASS_NAMES[NUM_CLS] = {"B_G", "B_1", "B_2", "B_3", "B_4", "B_5",
                                           "R_G", "R_1", "R_2", "R_3", "R_4", "R_5"};

class Logger : public nvinfer1::ILogger {
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING) std::cout << "[TRT] " << msg << std::endl;
    }
} gLogger;

#define CUDA_CHECK(x)                                                                    \
    do {                                                                                 \
        cudaError_t e = (x);                                                             \
        if (e != cudaSuccess) {                                                          \
            std::cerr << "CUDA error: " << cudaGetErrorString(e) << " @" << __LINE__     \
                      << std::endl;                                                      \
            std::exit(1);                                                                \
        }                                                                                \
    } while (0)

struct Detection {
    float kpts[8];  // 原图像素坐标 x1,y1,x2,y2,x3,y3,x4,y4
    float conf;
    int cls;
};

// ------------------------- engine 构建/加载 -------------------------

void buildEngine(const std::string& onnxPath, const std::string& enginePath) {
    std::cout << "构建 engine (首次运行, 需要几分钟)..." << std::endl;
    auto builder = nvinfer1::createInferBuilder(gLogger);
    auto network = builder->createNetworkV2(0);
    auto parser = nvonnxparser::createParser(*network, gLogger);
    if (!parser->parseFromFile(onnxPath.c_str(), int(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "ONNX 解析失败: " << onnxPath << std::endl;
        std::exit(1);
    }
    auto config = builder->createBuilderConfig();
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);
    auto serialized = builder->buildSerializedNetwork(*network, *config);
    if (!serialized) {
        std::cerr << "engine 构建失败" << std::endl;
        std::exit(1);
    }
    std::ofstream f(enginePath, std::ios::binary);
    f.write(static_cast<const char*>(serialized->data()), serialized->size());
    std::cout << "engine 已保存: " << enginePath << " (" << serialized->size() / 1024 / 1024
              << " MB)" << std::endl;
    delete serialized;
    delete config;
    delete parser;
    delete network;
    delete builder;
}

std::vector<char> loadFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::cerr << "无法打开: " << path << std::endl;
        std::exit(1);
    }
    size_t size = f.tellg();
    f.seekg(0);
    std::vector<char> buf(size);
    f.read(buf.data(), size);
    return buf;
}

// ------------------------- 前后处理 -------------------------

// letterbox: 等比缩放 + 灰边填充, 返回 gain 和 pad
cv::Mat letterbox(const cv::Mat& img, float& gain, float& padW, float& padH) {
    gain = std::min(float(INPUT_W) / img.cols, float(INPUT_H) / img.rows);
    int newW = int(std::round(img.cols * gain)), newH = int(std::round(img.rows * gain));
    padW = (INPUT_W - newW) / 2.0f;
    padH = (INPUT_H - newH) / 2.0f;
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(newW, newH));
    cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(out(cv::Rect(int(padW), int(padH), newW, newH)));
    return out;
}

// BGR HWC uint8 -> RGB CHW float32 / 255
void blobFromImage(const cv::Mat& img, float* blob) {
    const int area = INPUT_W * INPUT_H;
    for (int y = 0; y < INPUT_H; ++y) {
        const uchar* row = img.ptr<uchar>(y);
        for (int x = 0; x < INPUT_W; ++x) {
            int idx = y * INPUT_W + x;
            blob[0 * area + idx] = row[x * 3 + 2] / 255.0f;  // R
            blob[1 * area + idx] = row[x * 3 + 1] / 255.0f;  // G
            blob[2 * area + idx] = row[x * 3 + 0] / 255.0f;  // B
        }
    }
}

// 类别无关 NMS (外接框 IoU + IoMin 包含关系抑制, 与 python 端 polygon_nms 行为对齐)
std::vector<Detection> postprocess(const float* out, float confThres, float iouThres,
                                   float gain, float padW, float padH, int imgW, int imgH) {
    struct Cand {
        float kpts[8], score;
        int cls;
        float box[4];  // xyxy (640 尺度)
    };
    std::vector<Cand> cands;
    for (int i = 0; i < NUM_PRED; ++i) {
        const float* row = out + i * NUM_COLS;
        float obj = row[8];
        if (obj < confThres) continue;
        int best = 0;
        float bestCls = 0;
        for (int c = 0; c < NUM_CLS; ++c)
            if (row[9 + c] > bestCls) bestCls = row[9 + c], best = c;
        float score = obj * bestCls;  // conf = obj * cls
        if (score < confThres) continue;
        Cand cd;
        std::copy(row, row + 8, cd.kpts);
        cd.score = score;
        cd.cls = best;
        float x0 = 1e9f, y0 = 1e9f, x1 = -1e9f, y1 = -1e9f;
        for (int k = 0; k < 4; ++k) {
            x0 = std::min(x0, cd.kpts[k * 2]);
            x1 = std::max(x1, cd.kpts[k * 2]);
            y0 = std::min(y0, cd.kpts[k * 2 + 1]);
            y1 = std::max(y1, cd.kpts[k * 2 + 1]);
        }
        cd.box[0] = x0;
        cd.box[1] = y0;
        cd.box[2] = x1;
        cd.box[3] = y1;
        cands.push_back(cd);
    }
    std::sort(cands.begin(), cands.end(),
              [](const Cand& a, const Cand& b) { return a.score > b.score; });

    std::vector<Detection> dets;
    std::vector<bool> removed(cands.size(), false);
    const float iominThres = 0.6f;
    for (size_t i = 0; i < cands.size() && dets.size() < 50; ++i) {
        if (removed[i]) continue;
        const Cand& a = cands[i];
        for (size_t j = i + 1; j < cands.size(); ++j) {
            if (removed[j]) continue;
            const Cand& b = cands[j];
            float ix0 = std::max(a.box[0], b.box[0]), iy0 = std::max(a.box[1], b.box[1]);
            float ix1 = std::min(a.box[2], b.box[2]), iy1 = std::min(a.box[3], b.box[3]);
            float inter = std::max(0.f, ix1 - ix0) * std::max(0.f, iy1 - iy0);
            float areaA = (a.box[2] - a.box[0]) * (a.box[3] - a.box[1]);
            float areaB = (b.box[2] - b.box[0]) * (b.box[3] - b.box[1]);
            float iou = inter / (areaA + areaB - inter + 1e-7f);
            float iomin = inter / (std::min(areaA, areaB) + 1e-7f);
            if (iou > iouThres || iomin > iominThres) removed[j] = true;
        }
        Detection d;
        d.conf = a.score;
        d.cls = a.cls;
        for (int k = 0; k < 4; ++k) {  // 还原到原图坐标
            d.kpts[k * 2] = std::clamp((a.kpts[k * 2] - padW) / gain, 0.f, float(imgW - 1));
            d.kpts[k * 2 + 1] = std::clamp((a.kpts[k * 2 + 1] - padH) / gain, 0.f, float(imgH - 1));
        }
        dets.push_back(d);
    }
    return dets;
}

void drawDetections(cv::Mat& img, const std::vector<Detection>& dets) {
    for (const auto& d : dets) {
        // 蓝方画蓝色, 红方画红色 (BGR)
        cv::Scalar color = d.cls < 6 ? cv::Scalar(255, 128, 0) : cv::Scalar(0, 64, 255);
        std::vector<cv::Point> pts(4);
        for (int k = 0; k < 4; ++k)
            pts[k] = cv::Point(int(d.kpts[k * 2]), int(d.kpts[k * 2 + 1]));
        for (int k = 0; k < 4; ++k) cv::line(img, pts[k], pts[(k + 1) % 4], color, 2);
        for (int k = 0; k < 4; ++k) cv::circle(img, pts[k], 3, cv::Scalar(0, 255, 0), -1);
        char label[64];
        snprintf(label, sizeof(label), "%s %.2f", CLASS_NAMES[d.cls], d.conf);
        cv::Point org = pts[0] + cv::Point(0, -6);
        cv::putText(img, label, org, cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 0, 0), 3);
        cv::putText(img, label, org, cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1);
    }
}

// ------------------------- main -------------------------

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "用法: " << argv[0] << " <onnx> <engine> <images_dir> <out_dir> [conf] [iou]"
                  << std::endl;
        return 1;
    }
    std::string onnxPath = argv[1], enginePath = argv[2], imgDir = argv[3], outDir = argv[4];
    float confThres = argc > 5 ? std::stof(argv[5]) : 0.4f;
    float iouThres = argc > 6 ? std::stof(argv[6]) : 0.3f;

    if (!fs::exists(enginePath)) buildEngine(onnxPath, enginePath);

    auto engineData = loadFile(enginePath);
    auto runtime = nvinfer1::createInferRuntime(gLogger);
    auto engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
    auto context = engine->createExecutionContext();

    const size_t inSize = 3 * INPUT_H * INPUT_W, outSize = size_t(NUM_PRED) * NUM_COLS;
    void *dIn, *dOut;
    CUDA_CHECK(cudaMalloc(&dIn, inSize * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dOut, outSize * sizeof(__half)));
    context->setTensorAddress("images", dIn);
    context->setTensorAddress("output0", dOut);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // FP16 模型: host 侧 float 预处理后转 __half 上传, 输出 __half 转回 float
    std::vector<float> blob(inSize), output(outSize);
    std::vector<__half> blobH(inSize), outputH(outSize);

    // 收集图片
    std::vector<fs::path> images;
    for (auto& e : fs::directory_iterator(imgDir)) {
        auto ext = e.path().extension().string();
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp")
            images.push_back(e.path());
    }
    std::sort(images.begin(), images.end());
    fs::create_directories(outDir);
    std::cout << "共 " << images.size() << " 张图片, conf=" << confThres << " iou=" << iouThres
              << std::endl;

    // 预热
    for (int i = 0; i < 20; ++i) {
        context->enqueueV3(stream);
        cudaStreamSynchronize(stream);
    }

    std::vector<double> tPre, tInfer, tPost;
    size_t totalDets = 0;
    for (const auto& p : images) {
        cv::Mat img = cv::imread(p.string());
        if (img.empty()) continue;

        auto t0 = Clock::now();
        float gain, padW, padH;
        cv::Mat lb = letterbox(img, gain, padW, padH);
        blobFromImage(lb, blob.data());
        for (size_t k = 0; k < inSize; ++k) blobH[k] = __float2half(blob[k]);
        CUDA_CHECK(cudaMemcpyAsync(dIn, blobH.data(), inSize * sizeof(__half),
                                   cudaMemcpyHostToDevice, stream));
        auto t1 = Clock::now();

        context->enqueueV3(stream);
        CUDA_CHECK(cudaMemcpyAsync(outputH.data(), dOut, outSize * sizeof(__half),
                                   cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
        for (size_t k = 0; k < outSize; ++k) output[k] = __half2float(outputH[k]);
        auto t2 = Clock::now();

        auto dets = postprocess(output.data(), confThres, iouThres, gain, padW, padH,
                                img.cols, img.rows);
        auto t3 = Clock::now();

        totalDets += dets.size();
        drawDetections(img, dets);
        cv::imwrite((fs::path(outDir) / p.filename()).string(), img);

        tPre.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        tInfer.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
        tPost.push_back(std::chrono::duration<double, std::milli>(t3 - t2).count());
    }

    auto stat = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        return std::make_pair(mean, v[v.size() / 2]);
    };
    auto [preM, preMed] = stat(tPre);
    auto [infM, infMed] = stat(tInfer);
    auto [postM, postMed] = stat(tPost);
    double total = preM + infM + postM;

    std::cout << "\n===== 延迟统计 (" << images.size() << " 张, 均值/中位, ms) =====" << std::endl;
    printf("预处理:   %6.2f / %6.2f\n", preM, preMed);
    printf("推理(含拷贝): %6.2f / %6.2f\n", infM, infMed);
    printf("后处理:   %6.2f / %6.2f\n", postM, postMed);
    printf("端到端:   %6.2f ms  ->  %.0f FPS\n", total, 1000.0 / total);
    printf("检测总数: %zu (平均 %.2f /图)\n", totalDets, double(totalDets) / images.size());
    std::cout << "可视化结果已保存到: " << outDir << std::endl;

    cudaStreamDestroy(stream);
    cudaFree(dIn);
    cudaFree(dOut);
    delete context;
    delete engine;
    delete runtime;
    return 0;
}
