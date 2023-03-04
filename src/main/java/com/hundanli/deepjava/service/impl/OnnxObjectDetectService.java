package com.hundanli.deepjava.service.impl;

import ai.djl.Application;
import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.YoloV5Translator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import com.hundanli.deepjava.dto.DetectResultDTO;
import com.hundanli.deepjava.service.ObjectDetectService;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import lombok.extern.slf4j.Slf4j;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.springframework.stereotype.Service;
import org.springframework.util.StopWatch;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * @author hundanli
 * @version 1.0.0
 * @date 2023/3/4 13:07
 */
@Slf4j
@Service
public class OnnxObjectDetectService implements ObjectDetectService {

    private final String userDir = System.getProperty("user.dir");

    private ZooModel<Image, DetectedObjects> model;

    /**
     * 检测Image图像
     *
     * @param image 图片对象
     * @return 检测结果
     */
    @Override
    public DetectResultDTO detectObject(Image image) throws Exception {
        StopWatch stopWatch = new StopWatch();
        stopWatch.start("object detection");
        try (Predictor<Image, DetectedObjects> predictor = this.model.newPredictor()) {
            log.info("Start to detect objects from image.");
            DetectedObjects detectedObjects = predictor.predict(image);
            stopWatch.stop();
            long timeMillis = stopWatch.getTotalTimeMillis();
            log.info("Detection task cost time: {}ms", timeMillis);
            log.info("Detection result:{}", detectedObjects);
            Path boxImage = saveBoundingBoxImage(image, detectedObjects);
            DetectResultDTO detectResultDTO = new DetectResultDTO();
            detectResultDTO.setDetectedObjects(detectedObjects);
            detectResultDTO.setBoxImage(boxImage);
            return detectResultDTO;
        }
    }


    @PostConstruct
    void init() throws Exception {

        YoloV5Translator translator = YoloV5Translator.builder()
                .optOutputType(YoloV5Translator.YoloOutputType.AUTO)
                .optSynsetArtifactName("coco.names")
                .addTransform(new Resize(640, 640))
                .addTransform(new ToTensor()).build();
        Criteria<Image, DetectedObjects> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.OBJECT_DETECTION)
                        .setTypes(Image.class, DetectedObjects.class)
                        .optTranslator(translator)
                        .optDevice(Device.cpu())
                        .optModelPath(Paths.get(userDir, "src/main/resources/weights"))
                        .optModelName("yolov5n6")
                        .optEngine("OnnxRuntime")
                        .optProgress(new ProgressBar())
                        .build();

        this.model = criteria.loadModel();
        log.info("Loading Onnx Model, ModelName：{}, modelPath: {}", model.getName(), model.getModelPath());
    }

    @PreDestroy
    void destroy() {
        if (this.model != null) {
            this.model.close();
        }
    }

    private Path saveBoundingBoxImage(Image img, DetectedObjects detections)
            throws IOException {
        Path outputDir = Paths.get("target/output");
        Files.createDirectories(outputDir);
        List<DetectedObjects.DetectedObject> list = detections.items();
        int imageWidth = 640;
        int imageHeight = 640;
        List<String> classNames = new ArrayList<>();
        List<Double> probabilities = new ArrayList<>();
        List<BoundingBox> boundingBoxes = new ArrayList<>();
        for (DetectedObjects.DetectedObject result : list) {
            String className = result.getClassName();
            double probability = result.getProbability();
            BoundingBox box = result.getBoundingBox();

            Rectangle rectangle = box.getBounds();
            double x = (rectangle.getX() / imageWidth);
            double y = (rectangle.getY() / imageHeight);
            double width = rectangle.getWidth() / imageWidth;
            double height = rectangle.getHeight() / imageHeight;
            Rectangle rect = new Rectangle(x, y, width, height);
            classNames.add(className);
            probabilities.add(probability);
            boundingBoxes.add(rect);

        }
        DetectedObjects detectedObjects = new DetectedObjects(classNames, probabilities, boundingBoxes);
        img.drawBoundingBoxes(detectedObjects);
        long nanoTime = System.nanoTime();
        Path imagePath = outputDir.resolve(nanoTime + "-onnx-detected.png");
        // OpenJDK can't save jpg with alpha channel
        img.save(Files.newOutputStream(imagePath), "png");
        log.info("Detected objects image has been saved in: {}", imagePath);
        return imagePath;
    }

}
