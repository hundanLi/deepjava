package com.hundanli.deepjava.controller;

import com.hundanli.deepjava.dto.DetectResultDTO;
import com.hundanli.deepjava.service.ObjectDetectService;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.apache.commons.io.IOUtils;
import org.opencv.core.Mat;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.FileInputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Path;


/**
 * @author hundanli
 * @version 1.0.0
 * @date 2023/3/4 11:06
 */
@RestController
@RequestMapping("/objectDetect")
public class ObjectDetectController {

    @Autowired
    private ObjectDetectService objectDetectService;

    @PostMapping("/detectImage")
    public void detectObject(@RequestParam("file") MultipartFile imageFile,
                             HttpServletRequest request,
                             HttpServletResponse response) throws Exception {
        DetectResultDTO detectResultDTO = objectDetectService.detectObject(imageFile.getInputStream());
        Path boxImage = detectResultDTO.getBoxImage();
        try (OutputStream outputStream = response.getOutputStream();
             InputStream inputStream = new FileInputStream(boxImage.toFile())){
            IOUtils.copy(inputStream, outputStream);
        }
    }


}
