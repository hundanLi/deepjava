package com.hundanli.deepjava.service;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import com.hundanli.deepjava.dto.DetectResultDTO;

import java.io.*;

/**
 * @author hundanli
 * @version 1.0.0
 * @date 2023/3/4 11:08
 */
public interface ObjectDetectService {

    /**
     * 检测本地文件
     *
     * @param file 文件
     * @return 检测结果
     * @throws Exception 检测异常
     */
    default DetectResultDTO detectObject(File file) throws Exception {
        return detectObject(new FileInputStream(file));
    }


    /**
     * 检测文件流
     *
     * @param inputStream 文件流
     * @return 检测结果
     * @throws Exception 检测异常
     */
    default DetectResultDTO detectObject(InputStream inputStream) throws Exception {
        Image image = ImageFactory.getInstance().fromInputStream(inputStream);
        return detectObject(image);
    }


    /**
     * 检测Image图像
     *
     * @param image 图片对象
     * @return 检测结果
     * @throws Exception 检测异常
     */
    DetectResultDTO detectObject(Image image) throws Exception;


}
