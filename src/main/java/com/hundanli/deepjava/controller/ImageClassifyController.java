package com.hundanli.deepjava.controller;

import com.hundanli.deepjava.dto.ClassifyResultDTO;
import com.hundanli.deepjava.service.ImageClassifyService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

/**
 * @author hundanli
 * @version 1.0.0
 * @date 2023/3/4 11:07
 */
@RestController
@RequestMapping("imageClassify")
public class ImageClassifyController {

    @Autowired
    ImageClassifyService imageClassifyService;

    @PostMapping("/classify")
    public ClassifyResultDTO classifyImage(@RequestParam("file") MultipartFile file) throws Exception {
        return imageClassifyService.classify(file.getInputStream());
    }

}
