package com.hundanli.deepjava.controller;

import com.hundanli.deepjava.dto.DetectResultDTO;
import com.hundanli.deepjava.service.HumanFaceDetectService;
import jakarta.servlet.http.HttpServletResponse;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
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
 * @date 2023/3/4 11:07
 */
@RestController
@RequestMapping("/face")
public class FaceDetectController {

    @Autowired
    HumanFaceDetectService faceDetectService;

    @PostMapping("detectImage")
    public void detectImage(@RequestParam("file") MultipartFile multipartFile,
                            HttpServletResponse response) throws Exception {
        DetectResultDTO detectResultDTO = faceDetectService.detectFace(multipartFile.getInputStream());
        Path boxImage = detectResultDTO.getBoxImage();
        try (OutputStream outputStream = response.getOutputStream();
             InputStream inputStream = new FileInputStream(boxImage.toFile())) {
            IOUtils.copy(inputStream, outputStream);
        } finally {
            FileUtils.deleteQuietly(boxImage.toFile());
        }
    }

}
