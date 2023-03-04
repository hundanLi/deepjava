package com.hundanli.actionreg;

import ai.djl.Application;
import ai.djl.ModelException;
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
import ai.djl.translate.TranslateException;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.util.StopWatch;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

public final class ActionRecognitionTest {

    private static final Logger logger = LoggerFactory.getLogger(ActionRecognitionTest.class);

    private String userDir = System.getProperty("user.dir");

    private ActionRecognitionTest() {}

    @Test
    public void predict() throws IOException, ModelException, TranslateException {
        Path imageFile = Paths.get("src/main/resources/static/action_discus_throw.png");
        Image img = ImageFactory.getInstance().fromFile(imageFile);

        ImageClassificationTranslator translator = ImageClassificationTranslator.builder()
                .optSynsetArtifactName("classes.txt")
                .optApplySoftmax(true)
                .addTransform(new Resize(299, 299))
                .addTransform(new ToTensor())
                .addTransform(new Normalize(new float[]{0.485F, 0.456F, 0.406F}, new float[]{0.229F, 0.224F, 0.225F}))
                .build();

        Criteria<Image, Classifications> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.ACTION_RECOGNITION)
                        .setTypes(Image.class, Classifications.class)
//                        .optFilter("backbone", "inceptionv3")
//                        .optFilter("dataset", "ucf101")
                        .optTranslator(translator)
                        .optModelPath(Paths.get(userDir, "src/main/resources/weights/action"))
                        .optModelName("inceptionv3_ucf101")
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