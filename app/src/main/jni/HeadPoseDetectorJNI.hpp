#pragma once

#include <jni.h>

extern "C"
JNIEXPORT void JNICALL
Java_com_ghlab_mnnsample_jni_HeadPoseDetectorJNI_setModel(JNIEnv *env, jobject instance, jobject byteBuffer);

extern "C"
JNIEXPORT jobject JNICALL
Java_com_ghlab_mnnsample_jni_HeadPoseDetectorJNI_detectHeadPose(JNIEnv *env, jobject instance, jobject bitmap, jint format, jint rotation);

extern "C"
JNIEXPORT void JNICALL
Java_com_ghlab_mnnsample_jni_HeadPoseDetectorJNI_useQuantizedModel(JNIEnv *env, jobject instance, jboolean quantized);


