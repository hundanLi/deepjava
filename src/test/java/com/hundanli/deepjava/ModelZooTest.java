package com.hundanli.deepjava;

import ai.djl.Application;
import ai.djl.repository.Artifact;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.net.MalformedURLException;
import java.util.List;
import java.util.Map;

/**
 * @author hundanli
 * @version 1.0.0
 * @date 2023/2/4 17:07
 */
public class ModelZooTest {

    @Test
    void listModels() throws Exception {
        Map<Application, List<Artifact>> models = ModelZoo.listModels();
        models.forEach((app, artifacts) -> {
            System.out.println(app.getPath());
            if (!app.getPath().endsWith("image_classification")) {
                return;
            }
            for (Artifact artifact : artifacts) {
                System.out.println(artifact.getName() + ":" + artifact);
            }

            System.out.println("==============");
        });
    }
}
