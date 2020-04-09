#include "HeadPoseDetectorJNI.hpp"

#include "HeadPoseDetector.hpp"

#include <chrono>

#include <android/bitmap.h>
#include <android/log.h>
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, "HeadPoseDetector", __VA_ARGS__)

extern "C"
JNIEXPORT jobject JNICALL
Java_com_ghlab_mnnsample_jni_HeadPoseDetectorJNI_detectHeadPose(JNIEnv *env, jobject instance, jobject bitmap, jint format, jint rotation) {
    AndroidBitmapInfo info;
    void *pixels = 0;

    if (AndroidBitmap_getInfo(env, bitmap, &info) < 0 || AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0)
        return nullptr;

    AndroidBitmap_unlockPixels(env, bitmap);

    double yaw = 0;
    double pitch = 0;
    double roll = 0;

    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();

    if (!HeadPoseDetector::instance()->detect((const unsigned char *)pixels, info.width, info.height, info.stride, format, rotation, yaw, pitch, roll))
        return nullptr;

    jclass headPoseResultClass = env->FindClass("com/ghlab/mnnsample/jni/HeadPoseResult");
    jmethodID headPoseResultInitMId = env->GetMethodID(headPoseResultClass, "<init>", "(DDD)V");

    std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();

    //LOGD("duration : %lldms, yaw : %lf, pitch : %lf, roll : %lf", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(), yaw, pitch, roll);

    jobject result = env->NewObject(headPoseResultClass, headPoseResultInitMId, yaw, pitch, roll);

    env->DeleteLocalRef(headPoseResultClass);

    return result;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_ghlab_mnnsample_jni_HeadPoseDetectorJNI_useQuantizedModel(JNIEnv *env, jobject instance, jboolean quantized) {
    HeadPoseDetector::instance()->useQuantizedModel(quantized == JNI_TRUE);
}