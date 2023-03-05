package com.hundanli.classify;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.util.StopWatch;

import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * @author hundanli
 * @version 1.0.0
 * @date 2023/3/5 11:37
 */
public class YoloV5ClassifyTest {

    private final String userDir = System.getProperty("user.dir");

    private static final Logger logger = LoggerFactory.getLogger(YoloV5ClassifyTest.class);


    @Test
    void predict() throws Exception {
        Path imageFile = Paths.get("src/main/resources/static/Samoyed.png");
        Image img = ImageFactory.getInstance().fromFile(imageFile);

        logger.info("Supported Engines: ");
        Engine.getAllEngines().forEach(System.out::println);

        ImageClassificationTranslator translator = ImageClassificationTranslator.builder()
                .optSynsetArtifactName("coco.names")
                .optApplySoftmax(true)
                .addTransform(new Resize(224, 224))
                .addTransform(new ToTensor())
                .addTransform(new Normalize(new float[]{0.485F, 0.456F, 0.406F}, new float[]{0.229F, 0.224F, 0.225F}))
                .build();

        Criteria<Image, Classifications> criteria =
                Criteria.builder()
//                        .optApplication(Application.CV.IMAGE_CLASSIFICATION)
                        .setTypes(Image.class, Classifications.class)
                        .optTranslator(translator)
                        .optDevice(Device.cpu())
                        .optModelPath(Paths.get(userDir, "src/main/resources/weights/classify"))
                        .optModelName("yolov5s-cls")
                        .optEngine("OnnxRuntime")
                        .optProgress(new ProgressBar())
                        .build();

        try (ZooModel<Image, Classifications> model = criteria.loadModel()) {
            logger.info("Loading model of modelName：{}, modelPath: {}", model.getName(), model.getModelPath());
            StopWatch stopWatch = new StopWatch();
            stopWatch.start("image classify");
            try (Predictor<Image, Classifications> predictor = model.newPredictor()) {
                logger.info("Start to classify image: {}", imageFile);
                Classifications classifications = predictor.predict(img);
                stopWatch.stop();
                long timeMillis = stopWatch.getTotalTimeMillis();
                logger.info("Classify task cost time: {}ms", timeMillis);
                logger.info("Classify result:{}", classifications);
            }
        }

    }

}
