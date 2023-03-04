package com.hundanli.deepjava.controller;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;

/**
 * @author hundanli
 * @version 1.0.0
 * @date 2023/3/4 14:24
 */
@RestController
public class IndexPageController {

    @RequestMapping("/")
    public void index(HttpServletRequest request, HttpServletResponse response) throws IOException {
        response.sendRedirect("/index.html");

    }

}
