package com.hundanli.deepjava;

import ai.djl.Application;
import ai.djl.repository.Artifact;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * @author hundanli
 * @version 1.0.0
 * @date 2023/2/4 17:07
 */
public class ModelZooTest {

    @Test
    void listModels() throws ModelNotFoundException, IOException {
        Map<Application, List<Artifact>> models = ModelZoo.listModels();
        models.forEach((app, artifacts) -> {
            System.out.println(app.getPath());
            if (!app.getPath().endsWith("object_detection")) {
                return;
            }
            for (Artifact artifact : artifacts) {
                System.out.println(artifact.getName() + ": " + artifact.getResourceUri().toString());

            }

            System.out.println("==============");
        });
    }
}
