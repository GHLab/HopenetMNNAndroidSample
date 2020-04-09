package com.ghlab.mnnsample.jni;

public class HeadPoseResult {
    public double yaw;
    public double pitch;
    public double roll;

    public HeadPoseResult(double yaw, double pitch, double roll) {
        this.yaw = yaw;
        this.pitch = pitch;
        this.roll = roll;
    }
}
