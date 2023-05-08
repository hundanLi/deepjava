package com.hundanli.opencv;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.opencv.opencv_python3;
import org.junit.jupiter.api.Test;

import java.io.IOException;

import static org.bytedeco.cpython.global.python.*;

/**
 * @author hundanli
 * @version 1.0.0
 * @date 2023/3/17 21:09
 */
public class OpencvPythonTest {

    @Test
    void testLoader() throws IOException {
        Loader.load(opencv_python3.class);
        opencv_python3.cachePackage();
        Pointer program = Py_DecodeLocale(OpencvPythonTest.class.getSimpleName(), null);
        if (program == null) {
            System.err.println("Fatal error: cannot decode class name");
            System.exit(1);
        }
        Py_SetProgramName(program);  /* optional but recommended */
        Py_Initialize(cachePackages());
//        PyRun_SimpleString("from time import time,ctime\n"
//                + "print('Today is', ctime(time()))\n");
        PyRun_SimpleString("import cv2 as cv");
        if (Py_FinalizeEx() < 0) {
            System.exit(120);
        }
        PyMem_RawFree(program);
        System.exit(0);
    }

}
