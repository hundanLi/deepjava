package com.hundanli.actionreg;

import ai.djl.Application;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.util.StopWatch;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

public final class ActionRecognition {

    private static final Logger logger = LoggerFactory.getLogger(ActionRecognition.class);

    private ActionRecognition() {}

    @Test
    public void predict() throws IOException, ModelException, TranslateException {
        Path imageFile = Paths.get("src/test/resources/action_discus_throw.png");
        Image img = ImageFactory.getInstance().fromFile(imageFile);

        Criteria<Image, Classifications> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.ACTION_RECOGNITION)
                        .setTypes(Image.class, Classifications.class)
                        .optFilter("backbone", "inceptionv3")
                        .optFilter("dataset", "ucf101")
                        .optEngine("MXNet")
                        .optProgress(new ProgressBar())
                        .build();

        try (ZooModel<Image, Classifications> inception = criteria.loadModel()) {
            try (Predictor<Image, Classifications> action = inception.newPredictor()) {
                StopWatch stopWatch = new StopWatch();
                stopWatch.start();
                Classifications classifications = action.predict(img);
                logger.info("recognized actions: {}", classifications);
                stopWatch.stop();
                logger.info("cost time:{}ms", stopWatch.getTotalTimeMillis());
            }
        }
    }
}