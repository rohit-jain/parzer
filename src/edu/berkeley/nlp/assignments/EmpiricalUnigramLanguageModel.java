package edu.berkeley.nlp.assignments;

import edu.berkeley.nlp.langmodel.LanguageModel;
import edu.berkeley.nlp.util.Counter;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * A dummy language model -- uses empirical unigram counts, plus a single
 * ficticious count for unknown words.
 *
 * @author Dan Klein
 */
class EmpiricalUnigramLanguageModel implements LanguageModel {

  static final String STOP = "</S>";

  double total = 0.0;
  Counter<String> wordCounter = new Counter<String>();

  public double getWordProbability(List<String> sentence, int index) {
    String word = sentence.get(index);
    double count = wordCounter.getCount(word);
    if (count == 0) {
//      System.out.println("UNKNOWN WORD: "+sentence.get(index));
      return 1.0 / (total + 1.0);
    }
    return count / (total + 1.0);
  }

  public double getSentenceProbability(List<String> sentence) {
    List<String> stoppedSentence = new ArrayList<String>(sentence);
    stoppedSentence.add(STOP);
    double probability = 1.0;
    for (int index = 0; index < stoppedSentence.size(); index++) {
      probability *= getWordProbability(stoppedSentence, index);
    }
    return probability;
  }

  String generateWord() {
    double sample = Math.random();
    double sum = 0.0;
    for (String word : wordCounter.keySet()) {
      sum += wordCounter.getCount(word) / total;
      if (sum > sample) {
        return word;
      }
    }
    return "*UNKNOWN*";
  }

  public List<String> generateSentence() {
    List<String> sentence = new ArrayList<String>();
    String word = generateWord();
    while (!word.equals(STOP)) {
      sentence.add(word);
      word = generateWord();
    }
    return sentence;
  }

  public EmpiricalUnigramLanguageModel(Collection<List<String>> sentenceCollection) {
    for (List<String> sentence : sentenceCollection) {
      List<String> stoppedSentence = new ArrayList<String>(sentence);
      stoppedSentence.add(STOP);
      for (String word : stoppedSentence) {
        wordCounter.incrementCount(word, 1.0);
      }
    }
    total = wordCounter.totalCount();
  }
}
