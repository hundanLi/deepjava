package com.hundanli.deepjava.service;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import com.hundanli.deepjava.dto.ClassifyResultDTO;

import java.io.InputStream;

/**
 * @author hundanli
 * @version 1.0.0
 * @date 2023/3/4 11:09
 */
public interface ImageClassifyService {

    /**
     * 图片分类
     * @param inputStream 图片输入流
     * @return 结果
     * @throws Exception 分类异常
     */
    default ClassifyResultDTO classify(InputStream inputStream) throws Exception{
        return classify(ImageFactory.getInstance().fromInputStream(inputStream));
    }

    /**
     * 图片分类
     * @param image 图片
     * @return 结果
     * @throws Exception 分类异常
     */
    ClassifyResultDTO classify(Image image) throws Exception;

}
