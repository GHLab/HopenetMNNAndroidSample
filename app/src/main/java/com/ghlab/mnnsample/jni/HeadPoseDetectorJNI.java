package com.ghlab.mnnsample.jni;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

import java.nio.ByteBuffer;

public class HeadPoseDetectorJNI {
    native public static void setModel(ByteBuffer byteBuffer);
    native public static HeadPoseResult detectHeadPose(Bitmap bitmap, int format, int rotation);
    native public static void useQuantizedModel(boolean quantized);
}