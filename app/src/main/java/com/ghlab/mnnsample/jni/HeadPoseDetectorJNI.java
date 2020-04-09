package com.ghlab.mnnsample.jni;

import android.graphics.Bitmap;

public class HeadPoseDetectorJNI {
    native public static HeadPoseResult detectHeadPose(Bitmap bitmap, int format, int rotation);
    native public static void useQuantizedModel(boolean quantized);
}