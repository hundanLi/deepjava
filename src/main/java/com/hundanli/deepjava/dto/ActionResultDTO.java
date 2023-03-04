package com.hundanli.deepjava.dto;

import ai.djl.modality.Classifications;
import lombok.Data;

import java.util.ArrayList;
import java.util.List;


/**
 * @author hundanli
 * @version 1.0.0
 * @date 2023/3/4 16:44
 */
@Data
public class ActionResultDTO {

    private List<ActionProbability> classProbabilities;

    @Data
    public static class ActionProbability {
        private String actionName;
        private Double probability;

        public ActionProbability(String actionName, Double probability) {
            this.actionName = actionName;
            this.probability = probability;
        }
    }

    public ActionResultDTO(List<Classifications.Classification> classifications) {
        List<ActionProbability> classProbabilities = new ArrayList<>();
        for (Classifications.Classification classification : classifications) {
            ActionProbability probability = new ActionProbability(classification.getClassName(), classification.getProbability());
            classProbabilities.add(probability);
        }
        this.classProbabilities = classProbabilities;
    }
}
