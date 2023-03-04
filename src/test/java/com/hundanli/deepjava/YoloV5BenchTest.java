package com.hundanli.deepjava;

import ai.djl.Application;
import ai.djl.Device;
import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.YoloV5Translator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.opencv.opencv_java;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.util.StopWatch;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * An example of inference using an object detection model.
 *
 * <p>See this <a
 * href="https://github.com/deepjavalibrary/djl/blob/master/examples/docs/object_detection.md">doc</a>
 * for information about this example.
 */
public final class YoloV5BenchTest {

    private static final Logger logger = LoggerFactory.getLogger(YoloV5BenchTest.class);

    private final String userDir = System.getProperty("user.dir");

    @BeforeEach
    void init() throws Exception {
        Loader.load(opencv_java.class);
    }

    @Test
    public void predict() throws IOException, ModelException, TranslateException {
        Path imageFile = Paths.get("src/main/resources/static/dog_bike_car.jpg");
        Image img = ImageFactory.getInstance().fromFile(imageFile);

        Engine.getAllEngines().forEach(System.out::println);

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
                        .optModelPath(Paths.get(userDir,"src/main/resources/weights"))
                        .optModelName("yolov5n6")
//                        .optEngine("PyTorch")
                        .optEngine("OnnxRuntime")
                        .optProgress(new ProgressBar())
                        .build();

        try (ZooModel<Image, DetectedObjects> model = criteria.loadModel()) {
            logger.info("ModelName：{}, modelPath: {}", model.getName(), model.getModelPath());
            StopWatch stopWatch = new StopWatch();
            stopWatch.start("object detection");
            try (Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
                logger.info("Start to detect objects from image: {}", imageFile);
                DetectedObjects detectedObjects = predictor.predict(img);
                int count = 1;
                for (int i = 0; i < 100; i++) {
                    predictor.predict(img);
                    count++;
                }
                stopWatch.stop();
                long timeMillis = stopWatch.getTotalTimeMillis();
                logger.info("Detection task count={}, avg cost time: {} ms", count, (double) timeMillis / count);
                logger.info("Detection result:{}", detectedObjects);
                Path boxingPath = saveBoundingBoxImage(img, detectedObjects);
                HighGui.imshow("detectedObjects", Imgcodecs.imread(boxingPath.toFile().getAbsolutePath(), Imgcodecs.IMREAD_COLOR));
                HighGui.waitKey();
            }
        }
    }

    @Test
    public void qatPredict() throws IOException, ModelException, TranslateException {
        Path imageFile = Paths.get("src/test/resources/dog_bike_car.jpg");
        Image img = ImageFactory.getInstance().fromFile(imageFile);

        Engine.getAllEngines().forEach(System.out::println);

        YoloV5Translator translator = YoloV5Translator.builder()
                .optOutputType(YoloV5Translator.YoloOutputType.AUTO)
                .optSynsetArtifactName("weights/coco.names")
                .addTransform(new Resize(640, 640))
//                .addTransform(new ToTensor())
                .build();
        Criteria<Image, DetectedObjects> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.OBJECT_DETECTION)
                        .setTypes(Image.class, DetectedObjects.class)
                        .optTranslator(translator)
                        .optDevice(Device.cpu())
                        .optModelPath(Paths.get("D:\\developer\\project\\Learning\\deepjava\\src\\test\\resources"))
                        .optModelName("yolov5n6-qat")
//                        .optEngine("PyTorch")
                        .optEngine("OnnxRuntime")
                        .optProgress(new ProgressBar())
                        .build();

        try (ZooModel<Image, DetectedObjects> model = criteria.loadModel()) {
            logger.info("ModelName：{}, modelPath: {}", model.getName(), model.getModelPath());
            StopWatch stopWatch = new StopWatch();
            stopWatch.start("object detection");
            try (Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
                logger.info("Start to detect objects from image: {}", imageFile);
                DetectedObjects detectedObjects = predictor.predict(img);
                int count = 1;
                for (int i = 0; i < 100; i++) {
                    predictor.predict(img);
                    count++;
                }
                stopWatch.stop();
                long timeMillis = stopWatch.getTotalTimeMillis();
                logger.info("Detection task count={}, avg cost time: {} ms", count, (double) timeMillis / count);
                logger.info("Detection result:{}", detectedObjects);
                Path boxingPath = saveBoundingBoxImage(img, detectedObjects);
                HighGui.imshow("detectedObjects", Imgcodecs.imread(boxingPath.toFile().getAbsolutePath(), Imgcodecs.IMREAD_COLOR));
                HighGui.waitKey();
            }
        }
    }


    @Test
    public void batchPredict() throws IOException, ModelException, TranslateException {
        Path imageFile = Paths.get("src/test/resources/dog_bike_car.jpg");
        Image img = ImageFactory.getInstance().fromFile(imageFile);
        List<Image> imageList = new ArrayList<>();
        int batchSize = 16;
        for (int i = 0; i < batchSize; i++) {
            imageList.add(ImageFactory.getInstance().fromFile(imageFile));
        }
        Engine.getAllEngines().forEach(System.out::println);

        YoloV5Translator translator = YoloV5Translator.builder()
                .optOutputType(YoloV5Translator.YoloOutputType.AUTO)
                .optSynsetArtifactName("weights/coco.names")
                .addTransform(new Resize(640, 640))
                .addTransform(new ToTensor()).build();
        Criteria<Image, DetectedObjects> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.OBJECT_DETECTION)
                        .setTypes(Image.class, DetectedObjects.class)
                        .optTranslator(translator)
                        .optDevice(Device.cpu())
                        .optModelPath(Paths.get("D:\\developer\\project\\Learning\\deepjava\\src\\test\\resources"))
                        .optModelName("yolov5n6-dynamic")
//                        .optEngine("PyTorch")
                        .optEngine("OnnxRuntime")
                        .optProgress(new ProgressBar())
                        .build();

        try (ZooModel<Image, DetectedObjects> model = criteria.loadModel()) {
            logger.info("ModelName：{}, modelPath: {}", model.getName(), model.getModelPath());
            StopWatch stopWatch = new StopWatch();
            stopWatch.start("object detection");
            try (Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
                logger.info("Start to detect objects from image: {}", imageFile);
//                DetectedObjects detectedObjects = predictor.predict(img);
                int count = 1;
                List<DetectedObjects> detectedObjects = predictor.batchPredict(imageList);
                for (int i = 0; i < 8; i++) {
                    detectedObjects = predictor.batchPredict(imageList);
                    count++;
                }
                stopWatch.stop();
                long timeMillis = stopWatch.getTotalTimeMillis();
                logger.info("Detection task count={}, avg cost time: {} ms", batchSize * count, (double) timeMillis / (batchSize * count));
                logger.info("Detection result:{}", detectedObjects.get(0));
                Path boxingPath = saveBoundingBoxImage(img, detectedObjects.get(0));
                HighGui.imshow("detectedObjects", Imgcodecs.imread(boxingPath.toFile().getAbsolutePath(), Imgcodecs.IMREAD_COLOR));
                HighGui.waitKey();
            }
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

        Path imagePath = outputDir.resolve("bench-detected-dog_bike_car.png");
        // OpenJDK can't save jpg with alpha channel
        img.save(Files.newOutputStream(imagePath), "png");
        logger.info("Detected objects image has been saved in: {}", imagePath);
        return imagePath;
    }
}