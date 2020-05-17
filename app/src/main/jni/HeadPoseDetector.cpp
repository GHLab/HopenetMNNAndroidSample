#include "HeadPoseDetector.hpp"

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <iostream>

#include <android/bitmap.h>
#include <android/log.h>
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, "HeadPoseDetector", __VA_ARGS__)

#define kInputSize 128
#define kOutputSize 66

using namespace MNN::Express;

HeadPoseDetector::HeadPoseDetector()
{
    useQuantizedModel(false);
}

HeadPoseDetector::~HeadPoseDetector()
{
    m_interpreter->releaseModel();
    m_interpreter->releaseSession(m_session);
}

void HeadPoseDetector::init(const unsigned char *modelData, const int size)
{
    m_interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromBuffer(modelData, size));

    MNN::ScheduleConfig config;
    config.numThread = 2;
    config.type = MNN_FORWARD_CPU;
    MNN::BackendConfig backendConfig;
    backendConfig.precision = MNN::BackendConfig::PrecisionMode::Precision_Low;
    config.backendConfig = &backendConfig;

    m_session = m_interpreter->createSession(config);

    m_tensor = m_interpreter->getSessionInput(m_session, nullptr);
}

bool HeadPoseDetector::detect(const unsigned char *data, const int width, const int height, const int stride, const int format, const int rotation, /*out*/double &yaw, /*out*/double &pitch, /*out*/double &roll)
{
    m_interpreter->resizeTensor(m_tensor, {1, 3, kInputSize, kInputSize});
    m_interpreter->resizeSession(m_session);

    const float meanVals[3] = { 0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f };
    const float normVals[3] = { 1 / 0.229f / 255.f, 1 / 0.224f / 255.f, 1 / 0.225f / 255.f };

    std::shared_ptr<MNN::CV::ImageProcess> pretreat(
                MNN::CV::ImageProcess::create((MNN::CV::ImageFormat)format, MNN::CV::RGB, meanVals, 3, normVals, 3));

    MNN::CV::Matrix trans;
    trans.setIdentity();
    trans.postScale(1.0f / width, 1.0f / height);

    m_rotation = rotation;
    if (rotation != 0)
        trans.postRotate(rotation, 0.5f, 0.5f);

    trans.postScale(kInputSize, kInputSize);
    trans.invert(&trans);

    pretreat->setMatrix(trans);
    pretreat->convert(data, width, height, stride, m_tensor);

    m_interpreter->runSession(m_session);

    // hopenet_lite / hopenet_lite_fp16
//    const MNN::Tensor *yawTensor = m_interpreter->getSessionOutput(m_session, "616");
//    const MNN::Tensor *pitchTensor = m_interpreter->getSessionOutput(m_session, "617");
//    const MNN::Tensor *rollTensor = m_interpreter->getSessionOutput(m_session, "618");

    // hopenet_lite_mobilenetv2
//    const MNN::Tensor *yawTensor = m_interpreter->getSessionOutput(m_session, "477");
//    const MNN::Tensor *pitchTensor = m_interpreter->getSessionOutput(m_session, "478");
//    const MNN::Tensor *rollTensor = m_interpreter->getSessionOutput(m_session, "479");

    // hopenet_lite_mobilenetv3_0_75
    const MNN::Tensor *yawTensor = m_interpreter->getSessionOutput(m_session, "708");
    const MNN::Tensor *pitchTensor = m_interpreter->getSessionOutput(m_session, "716");
    const MNN::Tensor *rollTensor = m_interpreter->getSessionOutput(m_session, "724");

    yaw = __calcPoseValue(yawTensor);
    pitch = __calcPoseValue(pitchTensor);
    roll = rotation - __calcPoseValue(rollTensor);

    return true;
}

void HeadPoseDetector::useQuantizedModel(bool quantized)
{
    LOGD("useQuantizedModel : %d", quantized);

//    if (m_interpreter != nullptr)
//    {
//        m_interpreter->releaseModel();
//        m_interpreter->releaseSession(m_session);
//        m_interpreter.reset();
//    }
//
//    if (quantized)
//        m_interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromBuffer(hopenet_lite_quantized_mnn, sizeof(hopenet_lite_quantized_mnn)));
//    else
//        m_interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromBuffer(hopenet_lite_mnn, sizeof(hopenet_lite_mnn)));
//
//    MNN::ScheduleConfig config;
//    config.numThread = 1;
//    config.type = MNN_FORWARD_CPU;
//    MNN::BackendConfig backendConfig;
//    backendConfig.precision = MNN::BackendConfig::PrecisionMode::Precision_Low;
//    config.backendConfig = &backendConfig;
//
//    m_session = m_interpreter->createSession(config);
//
//    m_tensor = m_interpreter->getSessionInput(m_session, nullptr);
}

double HeadPoseDetector::__calcPoseValue(const MNN::Tensor *tensor)
{
    if (tensor == nullptr)
        return 0;

    tensor->getDimensionType();

    MNN::Tensor tensorHost(tensor, tensor->getDimensionType());

    tensor->copyToHostTensor(&tensorHost);

    std::vector<float> vals;

    for (int i = 0; i < kOutputSize; i++)
    {
        vals.push_back(tensorHost.host<float>()[i]);
    }

    auto input = _Input({1, 66}, NCHW);
    auto inputPtr = input->writeMap<float>();
    memcpy(inputPtr, vals.data(), vals.size() * sizeof(float));
    input->unMap();
    auto output = _Softmax(input);
    auto predicted = output->readMap<float>();

    double result = 0;
    for (int i = 0; i < kOutputSize; i++)
    {
        result += (predicted[i] * i);
    }

    return result * 3 - 99;
}