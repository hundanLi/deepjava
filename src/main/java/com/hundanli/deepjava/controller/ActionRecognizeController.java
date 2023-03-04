package com.hundanli.deepjava.controller;

import com.hundanli.deepjava.dto.ActionResultDTO;
import com.hundanli.deepjava.service.ActionRecognizeService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

/**
 * @author hundanli
 * @version 1.0.0
 * @date 2023/3/4 14:30
 */
@RestController
@RequestMapping("actionRecognize")
public class ActionRecognizeController {

    @Autowired
    ActionRecognizeService actionRecognizeService;

    @PostMapping("recognize")
    public ActionResultDTO actionRecognize(@RequestParam("file") MultipartFile file) throws Exception {
        return actionRecognizeService.recognizeAction(file.getInputStream());
    }

}
