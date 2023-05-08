package com.hundanli.deepjava.service.impl;

import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import com.hundanli.deepjava.dto.DetectResultDTO;
import com.hundanli.deepjava.service.HumanFaceDetectService;
import jakarta.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.time.StopWatch;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.opencv.opencv_java;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.FaceDetectorYN;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * @author zulong1.li
 */
@Slf4j
@Service
public final class YunetHumanFaceDetectService implements HumanFaceDetectService {

    private static final String FACE = "Face";

    private FaceDetectorYN detector;
    private final String userDir = System.getProperty("user.home");
    private Path weightPath;


    @Override
    public DetectResultDTO detectFace(InputStream inputStream) throws Exception {
        File tmpFile = new File("/tmp/video-detect/" + System.currentTimeMillis() + ".tmp");
        FileUtils.createParentDirectories(tmpFile);
        try (InputStream is = inputStream) {
            FileUtils.copyInputStreamToFile(is, tmpFile);
            return detectFace(tmpFile.getAbsolutePath());
        } finally {
            FileUtils.deleteQuietly(tmpFile);
        }

    }


    @Override
    public DetectResultDTO detectFace(String filepath) throws Exception {
        Mat img = Imgcodecs.imread(filepath, Imgcodecs.IMREAD_COLOR);
        return detectFace(img);
    }


    public DetectResultDTO detectFace(Mat img) throws IOException {
        Mat faces = new Mat();
        try {
            int height = img.height();
            int width = img.width();
            Size inputSize = new Size(width, height);
            StopWatch watch = new StopWatch();
            watch.start();
            detector.setInputSize(inputSize);
            detector.detect(img, faces);
            watch.stop();
            DetectedObjects detectedObjects = parseFaces(faces);
            log.info("Face detect cost time: {}ms", watch.getTime(TimeUnit.MILLISECONDS));
            Path boxImage = saveBoundingBoxImage(img, faces);
            DetectResultDTO detectResultDTO = new DetectResultDTO();
            detectResultDTO.setDetectedObjects(detectedObjects);
            detectResultDTO.setBoxImage(boxImage);
            return detectResultDTO;
        } finally {
            img.release();
            faces.release();
        }
    }


    private static Path saveBoundingBoxImage(Mat img, Mat faces)
            throws IOException {
        for (int i = 0; i < faces.rows(); i++) {
            Float x = faces.at(Float.class, i, 0).getV();
            Float y = faces.at(Float.class, i, 1).getV();
            Float boxWidth = faces.at(Float.class, i, 2).getV();
            Float boxHeight = faces.at(Float.class, i, 3).getV();
            Rect rect = new Rect(x.intValue(), y.intValue(), boxWidth.intValue(), boxHeight.intValue());
            Imgproc.rectangle(img, rect, new Scalar(0, 255, 0), 2);
        }
        Path outputDir = Paths.get("/tmp/video-detect/");
        Files.createDirectories(outputDir);
        Path imagePath = outputDir.resolve(System.nanoTime() + "_face_detected.jpg");
        Imgcodecs.imwrite(imagePath.toFile().getAbsolutePath(), img);
        log.info("Face detection result image has been saved in: {}", imagePath);
        return imagePath;
    }


    private DetectedObjects parseFaces(Mat faces) {
        List<String> classNames = new ArrayList<>();
        List<Double> probabilities = new ArrayList<>();
        List<BoundingBox> boundingBoxes = new ArrayList<>();
        for (int i = 0; i < faces.rows(); i++) {
            Float x = faces.at(Float.class, i, 0).getV();
            Float y = faces.at(Float.class, i, 1).getV();
            Float boxWidth = faces.at(Float.class, i, 2).getV();
            Float boxHeight = faces.at(Float.class, i, 3).getV();
            Float score = faces.at(Float.class, i, 14).getV();
            log.debug(String.format("Face %d, top-left coordinates: (%.0f, %.0f, box width: %.0f, box height: %.0f, score: %.2f", i, x, y, boxWidth, boxHeight, score));
            classNames.add(FACE);
            probabilities.add(score.doubleValue());
            boundingBoxes.add(new Rectangle(x, y, boxWidth, boxHeight));
        }
        return new DetectedObjects(classNames, probabilities, boundingBoxes);
    }

    @PostConstruct
    public void init() throws Exception {
        copyWeightFile();
        Loader.load(opencv_java.class);
        Size size = new Size(320, 320);
        this.detector = FaceDetectorYN.create(weightPath.toFile().getAbsolutePath(), "", size);

    }


    private void copyWeightFile() throws Exception {
        String weightFilePath = "weights/face/face_detection_yunet_2022mar.onnx";
        ClassPathResource classPathResource = new ClassPathResource(weightFilePath);
        Path destPath = Paths.get(userDir, ".video-detect-api", "face-detect", "yunet.onnx");
        Files.createDirectories(destPath.getParent());
        Files.copy(classPathResource.getInputStream(), destPath, StandardCopyOption.REPLACE_EXISTING);
        log.info("copy yunet.onnx file to: {}", destPath.toFile().getAbsolutePath());
        weightPath = destPath;
    }
}
