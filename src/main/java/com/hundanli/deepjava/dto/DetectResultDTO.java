package com.hundanli.deepjava.dto;

import ai.djl.modality.cv.output.DetectedObjects;
import lombok.Data;

import java.nio.file.Path;

/**
 * @author hundanli
 * @version 1.0.0
 * @date 2023/3/4 12:58
 */
@Data
public class DetectResultDTO {

    private DetectedObjects detectedObjects;
    private Path boxImage;

}
