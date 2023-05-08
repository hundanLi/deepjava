package com.hundanli.face;

import com.hundanli.deepjava.dto.DetectResultDTO;
import com.hundanli.deepjava.service.impl.YunetHumanFaceDetectService;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.opencv.opencv_java;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.opencv.core.Mat;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;

/**
 * @author hundanli
 * @version 1.0.0
 * @date 2023/4/11 22:08
 */
@Slf4j
public class FaceDetectTest {


    //    final String weightPath = "src/main/resources/weights/face/face_detection_yunet_2022mar.onnx";
    YunetHumanFaceDetectService detectService;


    @BeforeEach
    void beforeEach() throws Exception {
        this.detectService = new YunetHumanFaceDetectService();
        detectService.init();

    }


    @Test
    void testFaceDetect() throws Exception {
        String imgPath = "src/test/python/largest_selfie.jpg";
        DetectResultDTO resultDTO = detectService.detectFace(imgPath);
        Path boxImage = resultDTO.getBoxImage();
        Mat mat = Imgcodecs.imread(boxImage.toFile().getAbsolutePath(), Imgcodecs.IMREAD_UNCHANGED);
        HighGui.imshow("detected face", mat);
        HighGui.waitKey();

        FileUtils.deleteQuietly(boxImage.toFile());
    }


    @Test
    void detectFromCamera() throws IOException {
        Loader.load(opencv_java.class);
        VideoCapture capture = new VideoCapture(0);
        // 设置摄像头分辨率
        int width = 1280;
        int height = 720;
        capture.set(Videoio.CAP_PROP_FRAME_WIDTH, width);
        capture.set(Videoio.CAP_PROP_FRAME_HEIGHT, height);
        Mat img = new Mat();
        if (!capture.isOpened()) {
            log.error("Can't open camera");
            return;
        }
        HighGui.namedWindow("0");
//        HighGui.resizeWindow("0", 1920,1080);
        while (capture.read(img)) {
            if (img.empty()) {
                log.error("--(!) No captured frame -- Break!");
                break;
            }
            DetectResultDTO resultDTO = detectService.detectFace(img);
            if (resultDTO.getBoxImage() != null) {
                img.release();
                String filename = resultDTO.getBoxImage().toFile().getAbsolutePath();
                img = Imgcodecs.imread(filename, Imgcodecs.IMREAD_UNCHANGED);
            }
            HighGui.imshow("0", img);
            if (HighGui.waitKey(10) != -1) {
                break;// escape
            }
            img.release();
            if (resultDTO.getBoxImage() != null) {
                FileUtils.deleteQuietly(resultDTO.getBoxImage().toFile());
            }
        }
        img.release();
        capture.release();
    }
}
