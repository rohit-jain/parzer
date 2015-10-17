package edu.berkeley.nlp.assignments;

import java.util.*;
import java.io.*;

/**
 * @author Dan Klein
 */
class SpeechNBestList {
  List<String> correctSentence;
  List<List<String>> nBestSentences;
  Map<List<String>, Double> acousticScores;

  public List<String> getCorrectSentence() {
    return correctSentence;
  }

  public List<List<String>> getNBestSentences() {
    return nBestSentences;
  }

  public double getAcousticScore(List<String> sentence) {
    return acousticScores.get(sentence);
  }

  public SpeechNBestList(List<String> correctSentence, List<List<String>> nBestSentences, Map<List<String>, Double> acousticScores) {
    this.correctSentence = correctSentence;
    this.nBestSentences = nBestSentences;
    this.acousticScores = acousticScores;
  }

  static class Reader {
    public static List<SpeechNBestList> readSpeechNBestLists(String path, Set vocabulary) throws IOException {
      List<SpeechNBestList> speechNBestLists = new ArrayList<SpeechNBestList>();
      BufferedReader correctSentenceReader = open(path + "/REF.HUB1");
      Map<String, List<String>> correctSentenceMap = readCorrectSentences(correctSentenceReader);
      List<String> prefixList = getPrefixes(path);
      for (String prefix : prefixList) {
        BufferedReader wordReader = open(path + "/" + prefix);
        BufferedReader scoreReader = open(path + "/" + prefix + ".acc");
        List<String> correctSentence = correctSentenceMap.get(prefix);
        SpeechNBestList speechNBestList = buildSpeechNBestList(correctSentence, wordReader, scoreReader, vocabulary);
        if (speechNBestList != null)
          speechNBestLists.add(speechNBestList);
        wordReader.close();
        scoreReader.close();
      }
      correctSentenceReader.close();
      return speechNBestLists;
    }

    private static SpeechNBestList buildSpeechNBestList(List<String> correctSentence, BufferedReader wordReader, BufferedReader scoreReader, Set vocabulary) throws IOException {
      List<Double> scoreList = readScores(scoreReader);
      List<List<String>> sentenceList = readSentences(wordReader);
      List<List<String>> uniqueSentenceList = new ArrayList<List<String>>();
      Map<List<String>, Double> sentencesToScores = new HashMap<List<String>, Double>();
      List<String> tokenizedCorrectSentence = null;
      for (int i = 0; i < sentenceList.size(); i++) {
        List<String> sentence = sentenceList.get(i);
        if (! inVocabulary(sentence, vocabulary)) // && i < sentenceList.size()-1)
          continue;
        Double score = scoreList.get(i);
        Double bestScoreForSentence = sentencesToScores.get(sentence);
        if (!sentencesToScores.containsKey(sentence)) {
          uniqueSentenceList.add(sentence);
          if (equalsIgnoreSpaces(correctSentence, sentence)) {
            if (tokenizedCorrectSentence != null) {
              System.out.println("WARNING: SPEECH LATTICE ERROR");
            }
            tokenizedCorrectSentence = sentence;
          }
        }
        if (bestScoreForSentence == null || score > bestScoreForSentence) {
          sentencesToScores.put(sentence, score);
        }
      }
      if (uniqueSentenceList.isEmpty())
        return null;
      if (tokenizedCorrectSentence == null)
        return null;
      return new SpeechNBestList(tokenizedCorrectSentence, uniqueSentenceList, sentencesToScores);
    }

    private static boolean equalsIgnoreSpaces(List<String> sentence1, List<String> sentence2) {
      StringBuilder sb1 = new StringBuilder();
      StringBuilder sb2 = new StringBuilder();
      for (String word1 : sentence1) {
        sb1.append(word1);
      }
      for (String word2 : sentence2) {
        sb2.append(word2);
      }
      return sb1.toString().equalsIgnoreCase(sb2.toString());
    }

    private static boolean inVocabulary(List<String> sentence, Set vocabulary) {
      for (String word : sentence) {
        if (! vocabulary.contains(word))
          return false;
      }
      return true;
    }

    private static List<Double> readScores(BufferedReader scoreReader) throws IOException {
      List<Double> scoreList = new ArrayList<Double>();
      while (scoreReader.ready()) {
        String line = scoreReader.readLine();
        String[] scoreStrings = line.split("\\s+");
        double totalScore = 0.0;
        for (int i = 0; i < scoreStrings.length; i++) {
          String scoreString = scoreStrings[i];
          totalScore += Double.parseDouble(scoreString);
        }
        scoreList.add(totalScore);
      }
      return scoreList;
    }

    private static List<List<String>> readSentences(BufferedReader wordReader) throws IOException {
      List<List<String>> sentenceList = new ArrayList<List<String>>();
      while (wordReader.ready()) {
        String line = wordReader.readLine();
        String[] words = line.split("\\s+");
        List<String> sentence = new ArrayList<String>();
        for (int i = 0; i < words.length; i++) {
          String word = words[i];
          sentence.add(word.toLowerCase());
        }
        sentenceList.add(sentence);
      }
      return sentenceList;
    }

    private static List<String> getPrefixes(String path) {
      Set<String> prefixSet = new HashSet<String>();
      List<String> prefixList = new ArrayList<String>();
      File directory = new File(path);
      File[] files = directory.listFiles();
      for (int i = 0; i < files.length; i++) {
        File file = files[i];
        String fileName = file.getName();
        if (fileName.startsWith("REF"))
          continue;
        String prefix = fileName;
        int extensionIndex = fileName.lastIndexOf('.');
        if (extensionIndex > 0) {
          prefix = fileName.substring(0, extensionIndex);
        }
        if (prefixSet.contains(prefix))
          continue;
        prefixSet.add(prefix);
        prefixList.add(prefix);
      }
      return prefixList;
    }

    private static Map<String, List<String>> readCorrectSentences(BufferedReader reader) throws IOException {
      Map<String, List<String>> correctSentenceMap = new HashMap<String, List<String>>();
      while (reader.ready()) {
        String line = reader.readLine();
        String[] words = line.split("\\s+");
        List<String> sentence = new ArrayList<String>();
        for (int i = 0; i < words.length - 1; i++) {
          String word = words[i];
          sentence.add(word.toLowerCase());
        }
        String idToken = words[words.length - 1].toLowerCase();
        String sentenceID = idToken.substring(1, idToken.length() - 1);
        correctSentenceMap.put(sentenceID, sentence);
      }
      return correctSentenceMap;
    }

    private static BufferedReader open(String fileName) throws FileNotFoundException {
      return new BufferedReader(new FileReader(fileName));
    }
  }


}
