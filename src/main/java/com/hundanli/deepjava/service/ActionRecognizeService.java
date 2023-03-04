package com.hundanli.deepjava.service;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import com.hundanli.deepjava.dto.ActionResultDTO;

import java.io.InputStream;

/**
 * @author hundanli
 * @version 1.0.0
 * @date 2023/3/4 16:43
 */
public interface ActionRecognizeService {

    /**
     * 动作识别
     *
     * @param inputStream 图片输入流
     * @return 结果
     * @throws Exception 识别异常
     */
    default ActionResultDTO recognizeAction(InputStream inputStream) throws Exception {
        Image image = ImageFactory.getInstance().fromInputStream(inputStream);
        return recognizeAction(image);
    }

    /**
     * 图片动作识别
     *
     * @param image 图片
     * @return 结果
     * @throws Exception 识别异常
     */
    ActionResultDTO recognizeAction(Image image) throws Exception;

}
