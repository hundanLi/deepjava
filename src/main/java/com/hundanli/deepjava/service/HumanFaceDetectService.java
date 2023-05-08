package com.hundanli.deepjava.service;


import com.hundanli.deepjava.dto.DetectResultDTO;

import java.io.InputStream;

/**
 * @Title
 * @Description FaceDetectService
 * @Program video-service
 * @Author zulong1.li
 * @Version 1.0
 * @Date 2023-04-11 14:18
 * @Copyright Copyright (c) 2023 TCL Inc. All rights reserved
 **/
public interface HumanFaceDetectService {

    /**
     * 人脸检测
     * @param filePath 文件路径
     * @return 检测结果
     * @throws Exception 异常
     */
    DetectResultDTO detectFace(String filePath) throws Exception;

    /**
     * 人脸检测
     * @param inputStream 输入流
     * @return 检测结果
     * @throws Exception 异常
     */
    DetectResultDTO detectFace(InputStream inputStream) throws Exception;


}
