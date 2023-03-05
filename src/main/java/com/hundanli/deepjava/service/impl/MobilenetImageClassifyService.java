package com.hundanli.deepjava.service.impl;

import ai.djl.Application;
import ai.djl.Device;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.transform.CenterCrop;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Translator;
import com.hundanli.deepjava.dto.ClassifyResultDTO;
import com.hundanli.deepjava.service.ImageClassifyService;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.util.StopWatch;

import java.util.List;

/**
 * @author hundanli
 * @version 1.0.0
 * @date 2023/3/5 16:17
 */
@Slf4j
@Service
public class MobilenetImageClassifyService implements ImageClassifyService {

    private ZooModel<Image, Classifications> model;
    
    /**
     * 图片分类
     *
     * @param image 图片
     * @return 结果
     * @throws Exception 分类异常
     */
    @Override
    public ClassifyResultDTO classify(Image image) throws Exception {
        StopWatch stopWatch = new StopWatch();
        stopWatch.start("image classify");
        try (Predictor<Image, Classifications> predictor = model.newPredictor()) {
            log.info("Start to classify image");
            Classifications classifications = predictor.predict(image);
            stopWatch.stop();
            long timeMillis = stopWatch.getTotalTimeMillis();
            log.info("Image Classify task cost time: {}ms", timeMillis);
            log.info("Image Classify result:{}", classifications);
            List<Classifications.Classification> topK = classifications.topK();
            return new ClassifyResultDTO(topK);
        }

    }

    
    @PostConstruct
    void init() throws Exception {
        Translator<Image, Classifications> translator = ImageClassificationTranslator.builder()
                .addTransform(new Resize(256))
                .addTransform(new CenterCrop(224, 224))
                .addTransform(new ToTensor())
                .addTransform(new Normalize(
                        new float[] {0.485f, 0.456f, 0.406f},
                        new float[] {0.229f, 0.224f, 0.225f}))
                .optApplySoftmax(true)
                .build();
        Criteria<Image, Classifications> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.IMAGE_CLASSIFICATION)
                        .setTypes(Image.class, Classifications.class)
                        .optTranslator(translator)
                        .optDevice(Device.cpu())
                        .optArtifactId("ai.djl.mxnet:mobilenet:0.0.1")
                        .optFilter("flavor", "v3_large")
                        .optEngine("MXNet")
                        .optProgress(new ProgressBar())
                        .build();

        this.model = criteria.loadModel();
        log.info("Loading model of modelName：{}, modelPath: {}", model.getName(), model.getModelPath());

    }

    @PreDestroy
    void destroy() {
        if (model != null) {
            model.close();
        }
    }
}
