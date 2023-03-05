package com.hundanli.deepjava.service;

import java.io.InputStream;

/**
 * @author hundanli
 * @version 1.0.0
 * @date 2023/3/4 11:09
 */
public interface AudioClassifyService {

    /**
     * audio classify
     * @param inputStream 输入流
     */
    void classify(InputStream inputStream);

}
