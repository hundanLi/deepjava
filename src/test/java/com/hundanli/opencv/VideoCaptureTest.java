package com.hundanli.opencv;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.opencv.opencv_java;
import org.junit.jupiter.api.Test;
import org.opencv.core.Mat;
import org.opencv.highgui.HighGui;
import org.opencv.videoio.VideoCapture;

/**
 * @author hundanli
 * @version 1.0.0
 * @date 2023/2/27 23:18
 */
@Slf4j
public class VideoCaptureTest {


    @Test
    void videoCapture() {
        Loader.load(opencv_java.class);
        VideoCapture capture = new VideoCapture(0);
        Mat img = new Mat();
        if (!capture.isOpened()) {
            log.error("Can't open camera");
            return;
        }
        HighGui.namedWindow("0");
        while (capture.read(img)) {
            if (img.empty()) {
                log.error("--(!) No captured frame -- Break!");
                break;
            }
            HighGui.imshow("0", img);
            if (HighGui.waitKey(10) != -1) {
                break;// escape
            }
        }
        img.release();
        capture.release();

    }


}
