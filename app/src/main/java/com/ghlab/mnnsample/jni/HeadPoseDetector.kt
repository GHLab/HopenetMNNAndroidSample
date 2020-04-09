package com.ghlab.mnnsample.jni

import android.content.Context
import android.graphics.Bitmap

enum class ImageFormat(val value: Int) {
    RGBA(0),
    RGB(1),
    BGR(2),
    GRAY(3),
    BGRA(4),
    YUV_NV21(11),
    YUV_NV12(12)
}

enum class RotateFlags(val value: Int) {
    ROTATE_0(0),
    ROTATE_90_CLOCKWISE(90),
    ROTATE_180(180),
    ROTATE_90_COUNTER_CLOCKWISE(-90),
}

class HeadPoseDetector(context: Context) {
    private var context: Context? = null
    init {
        this.context = context
        System.loadLibrary("MNNSampleLib")
    }

    fun detectHeadPose(bitmap: Bitmap, format: ImageFormat = ImageFormat.RGBA, rotation: RotateFlags = RotateFlags.ROTATE_0): HeadPoseResult? {
        return HeadPoseDetectorJNI.detectHeadPose(bitmap, format.value, rotation.value)
    }

    fun useQuantizedModel(quantized: Boolean) {
        HeadPoseDetectorJNI.useQuantizedModel(quantized)
    }
}