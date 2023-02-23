package com.hundanli.opencv;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.time.StopWatch;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.opencv.opencv_java;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;



public class DarknetYoloV4Tests {

    @BeforeEach
    void loadLibrary() {
        Loader.load(opencv_java.class);
    }

    @Test
    void testImShow() {
        // Reading the Image from the file/ directory
        String imageLocation
                = "D:\\data\\filesaas\\sourceImage\\test.png";

        // Storing the image in a Matrix object
        // of Mat type
        Mat src = Imgcodecs.imread(imageLocation);

        // New matrix to store the final image
        // where the input image is supposed to be written
        Mat dst = new Mat();

        // Scaling the Image using Resize function
        Imgproc.resize(src, dst, new Size(0, 0), 0.7, 0.7,
                Imgproc.INTER_LANCZOS4);

        // Writing the image from src to destination
        Imgcodecs.imwrite("D:\\data\\filesaas\\lanczos4\\test-resized.png", dst);
        HighGui.imshow("resize", dst);
        HighGui.waitKey();
        // Display message to show that
        // image has been scaled
        System.out.println("Image Processed");
    }

    @Test
    void testImDetect() throws Exception{
        // 读取类别名称
        String[] names = new String[80];
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream("src/test/resources/coco.names")))) {
            for (int i = 0; i < names.length; i++) {
                names[i] = reader.readLine();
            }
        }

        StopWatch stopwatch = new StopWatch();

        // 定义对象
        Mat image = null;
        Mat out = null;
        MatOfInt indexes = null;
        MatOfRect2d boxes = null;
        MatOfFloat confidences = null;
        try {
            // 指定配置文件和模型文件加载网络
            stopwatch.start();
            String cfgFile = "src/test/resources/yolov4-tiny.cfg";
            String weights = "src/test/resources/yolov4-tiny.weights";
            Net net = Dnn.readNetFromDarknet(cfgFile, weights);
//            Net net = Dnn.readNet("src/test/resources/yolov5n.onnx");
//            Net net = Dnn.readNetFromTorch("src/test/resources/yolov5s.torchscript");
            if (net.empty()) {
                System.out.println("init net fail");
                return;
            }
            stopwatch.stop();
            System.out.println("load net and weights success, cost time: " + stopwatch.getTime(TimeUnit.MILLISECONDS));
            // 设置计算后台：如果电脑有GPU，可以指定为：DNN_BACKEND_CUDA
            net.setPreferableBackend(Dnn.DNN_BACKEND_OPENCV);
            // 指定为 CPU 模式
            net.setPreferableTarget(Dnn.DNN_TARGET_CPU);
            System.out.println("config net success");

            // 读取要被推理的图片
            String img_file = "src/test/resources/dog_bike_car.jpg";
            image = Imgcodecs.imread(img_file, Imgcodecs.IMREAD_COLOR);
            if (image.empty()) {
                System.out.println("read image fail");
                return;
            }
//            HighGui.imshow("input", image);
//            HighGui.waitKey();

            stopwatch.reset();
            stopwatch.start();
            // 图片预处理：将图片转换为 416 大小的图片，这个数值最好与配置文件的网络大小一致
            // 缩放因子大小，opencv 文档规定的：https://github.com/opencv/opencv/blob/master/samples/dnn/models.yml#L31
            double scale = 1 / 255D;
            Mat inputBlob = Dnn.blobFromImage(image, scale, new Size(416, 416), new Scalar(0), true, false, CvType.CV_32F);
//            Mat inputBlob = Dnn.blobFromImage(image, scale, new Size(640, 640), new Scalar(0), true, false, CvType.CV_32F);
            // 输入图片到网络中
            net.setInput(inputBlob);

            // 推理
            List<String> outLayersNames = net.getUnconnectedOutLayersNames();
            out = net.forward(outLayersNames.get(0));
            if (out.empty()) {
                System.out.println("forward result is null");
                return;
            }
            System.out.println("net forward success");
            stopwatch.stop();
            System.out.println("net forward cost time: " + stopwatch.getTime(TimeUnit.MILLISECONDS));
            // 处理 out 的结果集: 移除小的置信度数据和去重
            List<Rect2d> rect2dList = new ArrayList<>();
            List<Float> confList = new ArrayList<>();
            List<Integer> objIndexList = new ArrayList<>();
            // 每个 row 就是一个单元，cols 就是当前单元的预测信息
            for (int i = 0; i < out.rows(); i++) {
                int size = out.cols() * out.channels();
                float[] data = new float[size];
                // 将结果拷贝到 data 中，0 表示从索引0开始拷贝
                out.get(i, 0, data);
                float confidence = -1; // 置信度
                int objectClass = -1; // 类型索引
                // data中的前4个是box的数据，第5个是分数，后面是每个 classes 的置信度
                int objectClassConfStartIndex = 5;
                for (int j = objectClassConfStartIndex; j < out.cols(); j++) {
                    if (confidence < data[j]) {
                        // 记录本单元中最大的置信度及其类型索引
                        confidence = data[j];
                        objectClass = j - objectClassConfStartIndex;
                    }
                }
                if (confidence > 0.45) { // 置信度大于 0.45 的才记录
//                    System.out.println("max confidence: "+ confidence + " objClass: " + names[objectClass]);
                    // 计算中点、长宽、左下角点位
                    float centerX = data[0] * image.cols();
                    float centerY = data[1] * image.rows();
                    float width = data[2] * image.cols();
                    float height = data[3] * image.rows();
                    float leftBottomX = centerX - width / 2;
                    float leftBottomY = centerY - height / 2;

                    System.out.println("Class: " + names[objectClass]);
                    System.out.println("Confidence: " + confidence);
                    System.out.println("Box: " + leftBottomX + "," + leftBottomY + "," + width + "," + height);
                    // 记录box信息、置信度、类型索引
                    rect2dList.add(new Rect2d(leftBottomX, leftBottomY, width, height));
                    confList.add(confidence);
                    objIndexList.add(objectClass);
                }
            }
            if (rect2dList.isEmpty()) {
                System.out.println("not object");
                return;
            }
            // box 去重
            indexes = new MatOfInt();
            boxes = new MatOfRect2d(rect2dList.toArray(new Rect2d[0]));
            float[] confArr = new float[confList.size()];
            for (int i = 0; i < confList.size(); i++) {
                confArr[i] = confList.get(i);
            }
            confidences = new MatOfFloat(confArr);
            // NMS 算法去重
            Dnn.NMSBoxes(boxes, confidences, 0.5F, 0.45F, indexes);
            if (indexes.empty()) {
                System.out.println("indexes is empty");
                return;
            }
            // 对图片画框、输出每种类别出现的次数
            // 去重后的结果集
            int[] ints = indexes.toArray();
            int[] classesNumberList = new int[names.length];
            for (int i : ints) {
                // 与 names 的索引位置相对应
                Rect2d rect2d = rect2dList.get(i);
                Integer obj = objIndexList.get(i);
                classesNumberList[obj] += 1;
                // 将 box 信息画在图片上, Scalar 对象是 BGR 的顺序，与RGB顺序反着的。
                System.out.println("draw rectangle: " + names[obj]);
                Imgproc.rectangle(image, new Point(rect2d.x, rect2d.y), new Point(rect2d.x + rect2d.width, rect2d.y + rect2d.height),
                        new Scalar(0, 255, 0), 1);
            }

            Path path = Paths.get("D:\\tmp\\darknet\\outs", "out_" + System.currentTimeMillis() + ".jpg");
            FileUtils.forceMkdirParent(path.toFile());
            Imgcodecs.imwrite(path.toString(), image);
            HighGui.imshow("detect", image);
            HighGui.waitKey();
            for (int i = 0; i < names.length; i++) {
                if (classesNumberList[i] > 0) {
                    System.out.println(names[i] + ": " + classesNumberList[i]);
                }
            }
        } finally {
            // 释放资源
            if (image != null) {
                image.release();
            }
            if (out != null) {
                out.release();
            }
            if (indexes != null) {
                indexes.release();
            }
            if (boxes != null) {
                boxes.release();
            }
            if (confidences != null) {
                confidences.release();
            }
        }

    }

}
