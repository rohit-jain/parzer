package edu.berkeley.nlp.parser;

import edu.berkeley.nlp.ling.Tree;
import edu.berkeley.nlp.ling.Trees;

import java.util.*;
import java.io.PrintWriter;
import java.io.StringReader;

/**
 * Evaluates precision and recall for English Penn Treebank parse trees.  NOTE: Unlike the standard evaluation, multiplicity over each span is ignored.  Also, punction is NOT currently deleted properly (approximate hack), and other normalizations (like AVDP ~ PRT) are NOT done.
 *
 * @author Dan Klein
 */
public class EnglishPennTreebankParseEvaluator<L> {
    abstract static class AbstractEval<L> {

    protected String str = "";

    private int exact = 0;
    private int total = 0;

    private int correctEvents = 0;
    private int guessedEvents = 0;
    private int goldEvents = 0;

    abstract Set<Object> makeObjects(Tree<L> tree);

    public void evaluate(Tree<L> guess, Tree<L> gold) {
      evaluate(guess, gold, new PrintWriter(System.out, true));
    }

    /* evaluates precision and recall by calling makeObjects() to make a
     * set of structures for guess Tree and gold Tree, and compares them
     * with each other.  */
    public void evaluate(Tree<L> guess, Tree<L> gold, PrintWriter pw) {
      Set<Object> guessedSet = makeObjects(guess);
      Set<Object> goldSet = makeObjects(gold);
      Set<Object> correctSet = new HashSet<Object>();
      correctSet.addAll(goldSet);
      correctSet.retainAll(guessedSet);

      correctEvents += correctSet.size();
      guessedEvents += guessedSet.size();
      goldEvents += goldSet.size();

      int currentExact = 0;
      if (correctSet.size() == guessedSet.size() &&
          correctSet.size() == goldSet.size()) {
        exact++;
        currentExact = 1;
      }
      total++;

//      guess.pennPrint(pw);
//      gold.pennPrint(pw);
      displayPRF(str+" [Current] ", correctSet.size(), guessedSet.size(), goldSet.size(), currentExact, 1, pw);

    }

    private void displayPRF(String prefixStr, int correct, int guessed, int gold, int exact, int total, PrintWriter pw) {
      double precision = (guessed > 0 ? correct / (double) guessed : 1.0);
      double recall = (gold > 0 ? correct / (double) gold : 1.0);
      double f1 = (precision > 0.0 && recall > 0.0 ? 2.0 / (1.0 / precision + 1.0 / recall) : 0.0);

      double exactMatch = exact / (double) total;

      String displayStr = " P: " + ((int) (precision * 10000)) / 100.0 + " R: " + ((int) (recall * 10000)) / 100.0 + " F1: " + ((int) (f1 * 10000)) / 100.0 + " EX: "+((int) (exactMatch * 10000)) / 100.0 ;

      pw.println(prefixStr+displayStr);
    }

    public void display(boolean verbose) {
      display(verbose, new PrintWriter(System.out, true));
    }

    public void display(boolean verbose, PrintWriter pw) {
      displayPRF(str+" [Average] ", correctEvents, guessedEvents, goldEvents, exact, total, pw);
    }
  }

  static class LabeledConstituent<L> {
    L label;
    int start;
    int end;

    public L getLabel() {
      return label;
    }

    public int getStart() {
      return start;
    }

    public int getEnd() {
      return end;
    }

    public boolean equals(Object o) {
      if (this == o) return true;
      if (!(o instanceof LabeledConstituent)) return false;

      final LabeledConstituent labeledConstituent = (LabeledConstituent) o;

      if (end != labeledConstituent.end) return false;
      if (start != labeledConstituent.start) return false;
      if (label != null ? !label.equals(labeledConstituent.label) : labeledConstituent.label != null) return false;

      return true;
    }

    public int hashCode() {
      int result;
      result = (label != null ? label.hashCode() : 0);
      result = 29 * result + start;
      result = 29 * result + end;
      return result;
    }

    public String toString() {
      return label+"["+start+","+end+"]";
    }

    public LabeledConstituent(L label, int start, int end) {
      this.label = label;
      this.start = start;
      this.end = end;
    }
  }

  public static class LabeledConstituentEval<L> extends AbstractEval<L> {

    Set<L> labelsToIgnore;
    Set<L> punctuationTags;

    static <L> Tree<L> stripLeaves(Tree<L> tree) {
      if (tree.isLeaf())
        return null;
      if (tree.isPreTerminal())
        return new Tree<L>(tree.getLabel());
      List<Tree<L>> children = new ArrayList<Tree<L>>();
      for (Tree<L> child : tree.getChildren()) {
        children.add(stripLeaves(child));
      }
      return new Tree<L>(tree.getLabel(), children);
    }

    Set<Object> makeObjects(Tree<L> tree) {
      Tree<L> noLeafTree = stripLeaves(tree);
      Set<Object> set = new HashSet<Object>();
      addConstituents(noLeafTree, set, 0);
      return set;
    }

    private int addConstituents(Tree<L> tree, Set<Object> set, int start) {
      if (tree.isLeaf()) {
        if (punctuationTags.contains(tree.getLabel()))
          return 0;
        else
          return 1;
      }
      int end = start;
      for (Tree<L> child : tree.getChildren()) {
        int childSpan = addConstituents(child, set, end);
        end += childSpan;
      }
      L label = tree.getLabel();
      if (! labelsToIgnore.contains(label)) {
        set.add(new LabeledConstituent<L>(label, start, end));
      }
      return end - start;
    }


    public LabeledConstituentEval(Set<L> labelsToIgnore, Set<L> punctuationTags) {
      this.labelsToIgnore = labelsToIgnore;
      this.punctuationTags = punctuationTags;
    }

  }

  public static void main(String[] args) throws Throwable {
    Tree<String> goldTree = (new Trees.PennTreeReader(new StringReader("(ROOT (S (NP (DT the) (NN can)) (VP (VBD fell))))"))).next();
    Tree<String> guessedTree = (new Trees.PennTreeReader(new StringReader("(ROOT (S (NP (DT the)) (VP (MB can) (VP (VBD fell)))))"))).next();
    LabeledConstituentEval<String> eval = new LabeledConstituentEval<String>(Collections.singleton("ROOT"), new HashSet<String>());
    eval.evaluate(guessedTree, goldTree);
    eval.display(true);
  }
}
