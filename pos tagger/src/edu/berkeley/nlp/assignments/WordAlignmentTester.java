package edu.berkeley.nlp.assignments;

import edu.berkeley.nlp.util.*;
import edu.berkeley.nlp.io.IOUtils;

import java.util.*;
import java.io.*;

/**
 * Harness for testing word-level alignments.  The code is hard-wired for the
 * aligment source to be english and the alignment target to be french (recall
 * that's the direction for translating INTO english in the noisy channel
 * model).
 *
 * Your projects will implement several methods of word-to-word alignment.
 *
 * @author Dan Klein
 */
public class WordAlignmentTester {

  static final String ENGLISH_EXTENSION = "e";
  static final String FRENCH_EXTENSION = "f";

  /**
   * A holder for a pair of sentences, each a list of strings.  Sentences in
   * the test sets have integer IDs, as well, which are used to retreive the
   * gold standard alignments for those sentences.
   */
  public static class SentencePair {
    int sentenceID;
    String sourceFile;
    List<String> englishWords;
    List<String> frenchWords;

    public int getSentenceID() {
      return sentenceID;
    }

    public String getSourceFile() {
      return sourceFile;
    }

    public List<String> getEnglishWords() {
      return englishWords;
    }

    public List<String> getFrenchWords() {
      return frenchWords;
    }

    public String toString() {
      StringBuilder sb = new StringBuilder();
      for (int englishPosition = 0; englishPosition < englishWords.size(); englishPosition++) {
        String englishWord = englishWords.get(englishPosition);
        sb.append(englishPosition);
        sb.append(":");
        sb.append(englishWord);
        sb.append(" ");
      }
      sb.append("\n");
      for (int frenchPosition = 0; frenchPosition < frenchWords.size(); frenchPosition++) {
        String frenchWord = frenchWords.get(frenchPosition);
        sb.append(frenchPosition);
        sb.append(":");
        sb.append(frenchWord);
        sb.append(" ");
      }
      sb.append("\n");
      return sb.toString();
    }

    public SentencePair(int sentenceID, String sourceFile, List<String> englishWords, List<String> frenchWords) {
      this.sentenceID = sentenceID;
      this.sourceFile = sourceFile;
      this.englishWords = englishWords;
      this.frenchWords = frenchWords;
    }
  }

  /**
   * Alignments serve two purposes, both to indicate your system's guessed
   * alignment, and to hold the gold standard alignments.  Alignments map index
   * pairs to one of three values, unaligned, possibly aligned, and surely
   * aligned.  Your alignment guesses should only contain sure and unaligned
   * pairs, but the gold alignments contain possible pairs as well.
   *
   * To build an alignemnt, start with an empty one and use
   * addAlignment(i,j,true).  To display one, use the render method.
   */
  public static class Alignment {
    Set<Pair<Integer, Integer>> sureAlignments;
    Set<Pair<Integer, Integer>> possibleAlignments;

    public boolean containsSureAlignment(int englishPosition, int frenchPosition) {
      return sureAlignments.contains(new Pair<Integer, Integer>(englishPosition, frenchPosition));
    }

    public boolean containsPossibleAlignment(int englishPosition, int frenchPosition) {
      return possibleAlignments.contains(new Pair<Integer, Integer>(englishPosition, frenchPosition));
    }

    public void addAlignment(int englishPosition, int frenchPosition, boolean sure) {
      Pair<Integer, Integer> alignment = new Pair<Integer, Integer>(englishPosition, frenchPosition);
      if (sure)
        sureAlignments.add(alignment);
      possibleAlignments.add(alignment);
    }

    public Alignment() {
      sureAlignments = new HashSet<Pair<Integer, Integer>>();
      possibleAlignments = new HashSet<Pair<Integer, Integer>>();
    }

    public static String render(Alignment alignment, SentencePair sentencePair) {
      return render(alignment, alignment, sentencePair);
    }

    public static String render(Alignment reference, Alignment proposed, SentencePair sentencePair) {
      StringBuilder sb = new StringBuilder();
      for (int frenchPosition = 0; frenchPosition < sentencePair.getFrenchWords().size(); frenchPosition++) {
        for (int englishPosition = 0; englishPosition < sentencePair.getEnglishWords().size(); englishPosition++) {
          boolean sure = reference.containsSureAlignment(englishPosition, frenchPosition);
          boolean possible = reference.containsPossibleAlignment(englishPosition, frenchPosition);
          char proposedChar = ' ';
          if (proposed.containsSureAlignment(englishPosition, frenchPosition))
            proposedChar = '#';
          if (sure) {
            sb.append('[');
            sb.append(proposedChar);
            sb.append(']');
          } else {
            if (possible) {
              sb.append('(');
              sb.append(proposedChar);
              sb.append(')');
            } else {
              sb.append(' ');
              sb.append(proposedChar);
              sb.append(' ');
            }
          }
        }
        sb.append("| ");
        sb.append(sentencePair.getFrenchWords().get(frenchPosition));
        sb.append('\n');
      }
      for (int englishPosition = 0; englishPosition < sentencePair.getEnglishWords().size(); englishPosition++) {
        sb.append("---");
      }
      sb.append("'\n");
      boolean printed = true;
      int index = 0;
      while (printed) {
        printed = false;
        StringBuilder lineSB = new StringBuilder();
        for (int englishPosition = 0; englishPosition < sentencePair.getEnglishWords().size(); englishPosition++) {
          String englishWord = sentencePair.getEnglishWords().get(englishPosition);
          if (englishWord.length() > index) {
            printed = true;
            lineSB.append(' ');
            lineSB.append(englishWord.charAt(index));
            lineSB.append(' ');
          } else {
            lineSB.append("   ");
          }
        }
        index += 1;
        if (printed) {
          sb.append(lineSB);
          sb.append('\n');
        }
      }
      return sb.toString();
    }
  }

  /**
   * WordAligners have one method: alignSentencePair, which takes a sentence
   * pair and produces an alignment which specifies an english source for each
   * french word which is not aligned to "null".  Explicit alignment to
   * position -1 is equivalent to alignment to "null".
   */
  static interface WordAligner {
    Alignment alignSentencePair(SentencePair sentencePair);
  }

  /**
   * Simple alignment baseline which maps french positions to english positions.
   * If the french sentence is longer, all final word map to null.
   */
  static class BaselineWordAligner implements WordAligner {
    public Alignment alignSentencePair(SentencePair sentencePair) {
      Alignment alignment = new Alignment();
      int numFrenchWords = sentencePair.getFrenchWords().size();
      int numEnglishWords = sentencePair.getEnglishWords().size();
      for (int frenchPosition = 0; frenchPosition < numFrenchWords; frenchPosition++) {
        int englishPosition = frenchPosition;
        if (englishPosition >= numEnglishWords)
          englishPosition = -1;
        alignment.addAlignment(englishPosition, frenchPosition, true);
      }
      return alignment;
    }
  }

  public static void main(String[] args) {
    // Parse command line flags and arguments
    Map<String,String> argMap = CommandLineUtils.simpleCommandLineParser(args);

    // Set up default parameters and settings
    String basePath = ".";
    int maxTrainingSentences = 0;
    boolean verbose = false;
    String dataset = "mini";
    String model = "baseline";

    // Update defaults using command line specifications
    if (argMap.containsKey("-path")) {
      basePath = argMap.get("-path");
      System.out.println("Using base path: "+basePath);
    }
    if (argMap.containsKey("-sentences")) {
      maxTrainingSentences = Integer.parseInt(argMap.get("-sentences"));
      System.out.println("Using an additional "+maxTrainingSentences+" training sentences.");
    }
    if (argMap.containsKey("-data")) {
      dataset = argMap.get("-data");
      System.out.println("Running with data: "+dataset);
    } else {
      System.out.println("No data set specified.  Use -data [miniTest, validate, test].");
    }
    if (argMap.containsKey("-model")) {
      model = argMap.get("-model");
      System.out.println("Running with model: "+model);
    } else {
      System.out.println("No model specified.  Use -model modelname.");
    }
    if (argMap.containsKey("-verbose")) {
      verbose = true;
    }

    // Read appropriate training and testing sets.
    List<SentencePair> trainingSentencePairs = new ArrayList<SentencePair>();
    if (! dataset.equals("miniTest") && maxTrainingSentences > 0)
      trainingSentencePairs = readSentencePairs(basePath+"/training", maxTrainingSentences);
    List<SentencePair> testSentencePairs = new ArrayList<SentencePair>();
    Map<Integer,Alignment> testAlignments = new HashMap<Integer, Alignment>();
    if (dataset.equalsIgnoreCase("test")) {
      testSentencePairs = readSentencePairs(basePath+"/test", Integer.MAX_VALUE);
      testAlignments = readAlignments(basePath+"/answers/test.wa.nonullalign");
    } else if (dataset.equalsIgnoreCase("validate")) {
      testSentencePairs = readSentencePairs(basePath+"/trial", Integer.MAX_VALUE);
      testAlignments = readAlignments(basePath+"/trial/trial.wa");
    } else if (dataset.equalsIgnoreCase("miniTest")) {
      testSentencePairs = readSentencePairs(basePath+"/mini", Integer.MAX_VALUE);
      testAlignments = readAlignments(basePath+"/mini/mini.wa");
    } else {
      throw new RuntimeException("Bad data set mode: "+ dataset+", use test, validate, or miniTest.");
    }
    trainingSentencePairs.addAll(testSentencePairs);

    // Build model
    WordAligner wordAligner = null;
    if (model.equalsIgnoreCase("baseline")) {
      wordAligner = new BaselineWordAligner();
    }
    // TODO : build other alignment models

    // Test model
    test(wordAligner, testSentencePairs, testAlignments, verbose);
  }

  private static void test(WordAligner wordAligner, List<SentencePair> testSentencePairs, Map<Integer, Alignment> testAlignments, boolean verbose) {
    int proposedSureCount = 0;
    int proposedPossibleCount = 0;
    int sureCount = 0;
    int proposedCount = 0;
    for (SentencePair sentencePair : testSentencePairs) {
      Alignment proposedAlignment = wordAligner.alignSentencePair(sentencePair);
      Alignment referenceAlignment = testAlignments.get(sentencePair.getSentenceID());
      if (referenceAlignment == null)
        throw new RuntimeException("No reference alignment found for sentenceID "+sentencePair.getSentenceID());
      if (verbose) System.out.println("Alignment:\n"+Alignment.render(referenceAlignment,proposedAlignment,sentencePair));
      for (int frenchPosition = 0; frenchPosition < sentencePair.getFrenchWords().size(); frenchPosition++) {
        for (int englishPosition = 0; englishPosition < sentencePair.getEnglishWords().size(); englishPosition++) {
          boolean proposed = proposedAlignment.containsSureAlignment(englishPosition, frenchPosition);
          boolean sure = referenceAlignment.containsSureAlignment(englishPosition, frenchPosition);
          boolean possible = referenceAlignment.containsPossibleAlignment(englishPosition, frenchPosition);
          if (proposed && sure) proposedSureCount += 1;
          if (proposed && possible) proposedPossibleCount += 1;
          if (proposed) proposedCount += 1;
          if (sure) sureCount += 1;
        }
      }
    }
    System.out.println("Precision: "+proposedPossibleCount/(double)proposedCount);
    System.out.println("Recall: "+proposedSureCount/(double)sureCount);
    System.out.println("AER: "+(1.0-(proposedSureCount+proposedPossibleCount)/(double)(sureCount+proposedCount)));
  }


  // BELOW HERE IS IO CODE

  private static Map<Integer, Alignment> readAlignments(String fileName) {
    Map<Integer,Alignment> alignments = new HashMap<Integer, Alignment>();
    try {
      BufferedReader in = new BufferedReader(new FileReader(fileName));
      while (in.ready()) {
        String line = in.readLine();
        String[] words = line.split("\\s+");
        if (words.length != 4)
          throw new RuntimeException("Bad alignment file "+fileName+", bad line was "+line);
        Integer sentenceID = Integer.parseInt(words[0]);
        Integer englishPosition = Integer.parseInt(words[1])-1;
        Integer frenchPosition = Integer.parseInt(words[2])-1;
        String type = words[3];
        Alignment alignment = alignments.get(sentenceID);
        if (alignment == null) {
          alignment = new Alignment();
          alignments.put(sentenceID, alignment);
        }
        alignment.addAlignment(englishPosition, frenchPosition, type.equals("S"));
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    return alignments;
  }

  private static List<SentencePair> readSentencePairs(String path, int maxSentencePairs) {
    List<SentencePair> sentencePairs = new ArrayList<SentencePair>();
    List<String> baseFileNames = getBaseFileNames(path);
    for (String baseFileName : baseFileNames) {
      if (sentencePairs.size() >= maxSentencePairs)
        continue;
      sentencePairs.addAll(readSentencePairs(baseFileName));
    }
    return sentencePairs;
  }

  private static List<SentencePair> readSentencePairs(String baseFileName) {
    List<SentencePair> sentencePairs = new ArrayList<SentencePair>();
    String englishFileName = baseFileName + "." + ENGLISH_EXTENSION;
    String frenchFileName = baseFileName + "." + FRENCH_EXTENSION;
    try {
      BufferedReader englishIn = new BufferedReader(new FileReader(englishFileName));
      BufferedReader frenchIn = new BufferedReader(new FileReader(frenchFileName));
      while (englishIn.ready() && frenchIn.ready()) {
        String englishLine = englishIn.readLine();
        String frenchLine = frenchIn.readLine();
        Pair<Integer,List<String>> englishSentenceAndID = readSentence(englishLine);
        Pair<Integer,List<String>> frenchSentenceAndID = readSentence(frenchLine);
        if (! englishSentenceAndID.getFirst().equals(frenchSentenceAndID.getFirst()))
          throw new RuntimeException("Sentence ID confusion in file "+baseFileName+", lines were:\n\t"+englishLine+"\n\t"+frenchLine);
        sentencePairs.add(new SentencePair(englishSentenceAndID.getFirst(), baseFileName, englishSentenceAndID.getSecond(), frenchSentenceAndID.getSecond()));
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    return sentencePairs;
  }

  private static Pair<Integer, List<String>> readSentence(String line) {
    int id = -1;
    List<String> words = new ArrayList<String>();
    String[] tokens = line.split("\\s+");
    for (int i = 0; i < tokens.length; i++) {
      String token = tokens[i];
      if (token.equals("<s")) continue;
      if (token.equals("</s>")) continue;
      if (token.startsWith("snum=")) {
        String idString = token.substring(5,token.length()-1);
        id = Integer.parseInt(idString);
        continue;
      }
      words.add(token.intern());
    }
    return new Pair<Integer, List<String>>(id, words);
  }

  private static List<String> getBaseFileNames(String path) {
    List<File> englishFiles = IOUtils.getFilesUnder(path, new FileFilter() {
      public boolean accept(File pathname) {
        if (pathname.isDirectory())
          return true;
        String name = pathname.getName();
        return name.endsWith(ENGLISH_EXTENSION);
      }
    });
    List<String> baseFileNames = new ArrayList<String>();
    for (File englishFile : englishFiles) {
      String baseFileName = chop(englishFile.getAbsolutePath(), "."+ENGLISH_EXTENSION);
      baseFileNames.add(baseFileName);
    }
    return baseFileNames;
  }

  private static String chop(String name, String extension) {
    if (! name.endsWith(extension)) return name;
    return name.substring(0, name.length()-extension.length());
  }

}
