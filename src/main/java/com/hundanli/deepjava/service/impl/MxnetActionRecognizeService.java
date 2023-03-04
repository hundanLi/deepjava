package com.hundanli.deepjava.service.impl;

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
import com.hundanli.deepjava.dto.ActionResultDTO;
import com.hundanli.deepjava.service.ActionRecognizeService;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.util.StopWatch;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

/**
 * @author hundanli
 * @version 1.0.0
 * @date 2023/3/4 16:46
 */
@Slf4j
@Service
public class MxnetActionRecognizeService implements ActionRecognizeService {

    private ZooModel<Image, Classifications> model;

    @Override
    public ActionResultDTO recognizeAction(Image image) throws Exception {
        try (Predictor<Image, Classifications> action = model.newPredictor()) {
            StopWatch stopWatch = new StopWatch();
            stopWatch.start();
            Classifications classifications = action.predict(image);
            log.info("recognized actions: {}", classifications);
            stopWatch.stop();
            log.info("recognized action cost time:{}ms", stopWatch.getTotalTimeMillis());
            List<Classifications.Classification> classificationList = classifications.topK();
            return new ActionResultDTO(classificationList);
        }
    }


    @PostConstruct
    void init() throws Exception {
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
                        .optModelPath(Paths.get(System.getProperty("user.dir"), "src/main/resources/weights/action"))
                        .optModelName("inceptionv3_ucf101")
                        .optEngine("MXNet")
                        .optProgress(new ProgressBar())
                        .build();

        this.model = criteria.loadModel();
        log.info("Loading Onnx Model, ModelName：{}, modelPath: {}", model.getName(), model.getModelPath());

    }

    @PreDestroy
    void destroy() {
        if (model != null) {
            model.close();
        }
    }

}
