package edu.berkeley.nlp.assignments;

import edu.berkeley.nlp.classify.*;
import edu.berkeley.nlp.util.Counter;

import java.util.List;

/**
 * Baseline classifier which always chooses the class seen most frequently in training.
 *
 * @author Dan Klein
 */
public class MostFrequentLabelClassifier<I,L> implements ProbabilisticClassifier<I,L> {

  public static class Factory<I,L> implements ProbabilisticClassifierFactory<I,L> {

    public ProbabilisticClassifier<I,L> trainClassifier(List<LabeledInstance<I,L>> trainingData) {
      Counter<L> labels = new Counter<L>();
      for (LabeledInstance<I,L> datum : trainingData) {
        labels.incrementCount(datum.getLabel(), 1.0);
      }
      return new MostFrequentLabelClassifier<I,L>(labels);
    }

  }

  Counter<L> labels;

  public Counter<L> getProbabilities(I input) {
    Counter<L> counter = new Counter<L>();
    counter.incrementAll(labels);
    counter.normalize();
    return counter;
  }

  public L getLabel(I input) {
    return labels.argMax();
  }

  public MostFrequentLabelClassifier(Counter<L> labels) {
    this.labels = labels;
  }

  public MostFrequentLabelClassifier(L label) {
    this.labels = new Counter<L>();
    labels.incrementCount(label, 1.0);
  }
}
