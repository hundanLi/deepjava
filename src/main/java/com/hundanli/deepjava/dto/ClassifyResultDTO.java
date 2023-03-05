package com.hundanli.deepjava.dto;

import ai.djl.modality.Classifications;
import lombok.Data;

import java.util.ArrayList;
import java.util.List;

/**
 * @author hundanli
 * @version 1.0.0
 * @date 2023/3/5 16:12
 */
@Data
public class ClassifyResultDTO {


    private List<ClassProbability> classProbabilities;


    public ClassifyResultDTO(List<Classifications.Classification> classifications) {
        List<ClassProbability> classProbabilities = new ArrayList<>();
        for (Classifications.Classification classification : classifications) {
            ClassProbability probability = new ClassProbability(classification.getClassName(), classification.getProbability());
            classProbabilities.add(probability);
        }
        this.classProbabilities = classProbabilities;
    }


    @Data
    public static class ClassProbability {
        private String className;
        private Double probability;

        public ClassProbability(String className, Double probability) {
            this.className = className;
            this.probability = probability;
        }
    }


}
