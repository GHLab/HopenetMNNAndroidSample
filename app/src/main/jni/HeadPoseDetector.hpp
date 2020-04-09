#pragma once

#include <memory>
#include <unordered_map>

#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

class HeadPoseDetector
{
public:
    HeadPoseDetector(HeadPoseDetector const&) = delete;
    HeadPoseDetector& operator=(HeadPoseDetector const&) = delete;
    
    static std::shared_ptr<HeadPoseDetector> instance()
    {
        static std::shared_ptr<HeadPoseDetector> s { new HeadPoseDetector };
        return s;
    }
    ~HeadPoseDetector();
    
private:
    HeadPoseDetector();
    
public:
    bool detect(const unsigned char *data, const int width, const int height, const int stride, const int format, const int rotation, /*out*/double &yaw, /*out*/double &pitch, /*out*/double &roll);
    void useQuantizedModel(bool quantized);

private:
    double __calcPoseValue(const MNN::Tensor *tensor);
    
private:
    std::shared_ptr<MNN::Interpreter> m_interpreter;
    MNN::Session *m_session = nullptr;
    MNN::Tensor *m_tensor = nullptr;
    
    int m_rotation;
};
