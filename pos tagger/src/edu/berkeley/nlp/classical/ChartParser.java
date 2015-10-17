package edu.berkeley.nlp.classical;

import edu.berkeley.nlp.ling.Tree;
import edu.berkeley.nlp.ling.Trees;
import edu.berkeley.nlp.util.CollectionUtils;
import edu.berkeley.nlp.util.Filter;

import java.util.*;
import java.util.regex.Pattern;
import java.util.regex.Matcher;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.FileReader;
import java.io.FileNotFoundException;

/**
 * @author Dan Klein
 */
public class ChartParser {

  static boolean verbose = false;

  static class Lexicon {
    Map<String,List<String>> wordToTags = new HashMap<String, List<String>>();
    public List<String> getTags(String word) {
      return wordToTags.get(word);
    }
    public String toString() {
      StringBuilder sb = new StringBuilder();
      for (String word: CollectionUtils.sort(wordToTags.keySet())) {
        List<String> tags = wordToTags.get(word);
        sb.append(word);
        sb.append(" :");
        for (String tag: CollectionUtils.sort(tags)) {
          sb.append(" ");
          sb.append(tag);
        }
      }
      return sb.toString();
    }
    public Lexicon(BufferedReader in) throws IOException {
      Pattern linePattern = Pattern.compile("^\\s*(\\S+)\\s*:\\s*(.*\\S+)\\s*$");
      while (in.ready()) {
        String line = in.readLine();
        Matcher matcher = linePattern.matcher(line);
        if (!matcher.matches()) throw new RuntimeException("Bad line in lexicon: "+line);
        String word = matcher.group(1);
        String tags = matcher.group(2);
        String[] tagArray = tags.split("\\s+");
        List<String> tagList = Arrays.asList(tagArray);
        wordToTags.put(word,tagList);
      }
    }
  }
  static class Grammar {
    Map<String, List<BinaryRule>> binaryRulesByLeftChild = new HashMap<String, List<BinaryRule>>();
    Map<String, List<BinaryRule>> binaryRulesByRightChild = new HashMap<String, List<BinaryRule>>();
    Map<String, List<UnaryRule>> unaryRulesByChild = new HashMap<String, List<UnaryRule>>();

    public List<BinaryRule> getBinaryRulesByLeftChild(String leftChild) {
      return CollectionUtils.getValueList(binaryRulesByLeftChild, leftChild);
    }

    public List<BinaryRule> getBinaryRulesByRightChild(String rightChild) {
      return CollectionUtils.getValueList(binaryRulesByRightChild, rightChild);
    }

    public List<UnaryRule> getUnaryRulesByChild(String child) {
      return CollectionUtils.getValueList(unaryRulesByChild, child);
    }

    public String toString() {
      StringBuilder sb = new StringBuilder();
      List<String> ruleStrings = new ArrayList<String>();
      for (String leftChild : binaryRulesByLeftChild.keySet()) {
        for (BinaryRule binaryRule : getBinaryRulesByLeftChild(leftChild)) {
          ruleStrings.add(binaryRule.toString());
        }
      }
      for (String child : unaryRulesByChild.keySet()) {
        for (UnaryRule unaryRule : getUnaryRulesByChild(child)) {
          ruleStrings.add(unaryRule.toString());
        }
      }
      for (String ruleString : CollectionUtils.sort(ruleStrings)) {
        sb.append(ruleString);
        sb.append("\n");
      }
      return sb.toString();
    }

    public Grammar(BufferedReader in) throws IOException {
      Pattern linePattern = Pattern.compile("^\\s*(\\S+)\\s*-*>\\s*(.*\\S+)\\s*$");
      while (in.ready()) {
        String line = in.readLine();
        Matcher matcher = linePattern.matcher(line);
        if (!matcher.matches()) throw new RuntimeException("Bad line in grammar: "+line);
        String parent = matcher.group(1);
        String children = matcher.group(2);
        String[] childArray = children.split("\\s+");
        if (childArray.length == 1) {
          // unary rule
          addUnary(new UnaryRule(parent, childArray[0]));
        } else if (childArray.length == 2) {
          // binary rule
          addBinary(new BinaryRule(parent, childArray[0], childArray[1]));
        } else {
          // left-trie binarize an n-ary rule
          StringBuilder intermediateSymbol = new StringBuilder("@NP->");
          String lastRight = childArray[0];
          intermediateSymbol.append(lastRight);
          int pos = 1;
          while (pos < childArray.length) {
            String leftChild = intermediateSymbol.toString();
            if (pos == 1) leftChild = childArray[0];
            String rightChild = childArray[pos];
            intermediateSymbol.append("_");
            intermediateSymbol.append(rightChild);
            String intermediateParent = intermediateSymbol.toString();
            if (pos == childArray.length-1) intermediateParent = parent;
            addBinary(new BinaryRule(intermediateParent, leftChild, rightChild));
            pos++;
          }
        }
      }
    }

    private void addBinary(BinaryRule binaryRule) {
      CollectionUtils.addToValueList(binaryRulesByLeftChild, binaryRule.getLeftChild(), binaryRule);
      CollectionUtils.addToValueList(binaryRulesByRightChild, binaryRule.getRightChild(), binaryRule);
    }

    private void addUnary(UnaryRule unaryRule) {
      CollectionUtils.addToValueList(unaryRulesByChild, unaryRule.getChild(), unaryRule);
    }
  }
  static class BinaryRule {
    String parent;
    String leftChild;
    String rightChild;

    public String getParent() {
      return parent;
    }

    public String getLeftChild() {
      return leftChild;
    }

    public String getRightChild() {
      return rightChild;
    }

    public boolean equals(Object o) {
      if (this == o) return true;
      if (!(o instanceof BinaryRule)) return false;

      final BinaryRule binaryRule = (BinaryRule) o;

      if (leftChild != null ? !leftChild.equals(binaryRule.leftChild) : binaryRule.leftChild != null) return false;
      if (parent != null ? !parent.equals(binaryRule.parent) : binaryRule.parent != null) return false;
      if (rightChild != null ? !rightChild.equals(binaryRule.rightChild) : binaryRule.rightChild != null) return false;

      return true;
    }

    public int hashCode() {
      int result;
      result = (parent != null ? parent.hashCode() : 0);
      result = 29 * result + (leftChild != null ? leftChild.hashCode() : 0);
      result = 29 * result + (rightChild != null ? rightChild.hashCode() : 0);
      return result;
    }

    public String toString() {
      return parent + " -> " + leftChild + " " + rightChild;
    }

    public BinaryRule(String parent, String leftChild, String rightChild) {
      this.parent = parent;
      this.leftChild = leftChild;
      this.rightChild = rightChild;
    }
  }
  static class UnaryRule {
    String parent;
    String child;

    public String getParent() {
      return parent;
    }

    public String getChild() {
      return child;
    }

    public boolean equals(Object o) {
      if (this == o) return true;
      if (!(o instanceof UnaryRule)) return false;

      final UnaryRule unaryRule = (UnaryRule) o;

      if (child != null ? !child.equals(unaryRule.child) : unaryRule.child != null) return false;
      if (parent != null ? !parent.equals(unaryRule.parent) : unaryRule.parent != null) return false;

      return true;
    }

    public int hashCode() {
      int result;
      result = (parent != null ? parent.hashCode() : 0);
      result = 29 * result + (child != null ? child.hashCode() : 0);
      return result;
    }

    public String toString() {
      return parent + " -> "+child;
    }

    public UnaryRule(String parent, String child) {
      this.parent = parent;
      this.child = child;
    }
  }

  static class Edge {
    static class Backtrace {}
    static class WordBacktrace extends Backtrace {
      String word;
      public String getWord() {
        return word;
      }
      public WordBacktrace(String word) {
        this.word = word;
      }
    }
    static class UnaryBacktrace extends Backtrace {
      Edge childEdge;
      public Edge getChildEdge() {
        return childEdge;
      }
      public UnaryBacktrace(Edge child) {
        this.childEdge = child;
      }
    }
    static class BinaryBacktrace extends Backtrace {
      Edge leftEdge;
      Edge rightEdge;
      public Edge getLeftEdge() {
        return leftEdge;
      }
      public Edge getRightEdge() {
        return rightEdge;
      }
      public BinaryBacktrace(Edge leftChild, Edge rightChild) {
        this.leftEdge = leftChild;
        this.rightEdge = rightChild;
      }
    }

    String label;
    int start;
    int end;
    boolean discovered;
    List<Backtrace> backtraces;

    public String getLabel() {
      return label;
    }

    public int getStart() {
      return start;
    }

    public int getEnd() {
      return end;
    }

    public boolean isDiscovered() {
      return discovered;
    }

    public void setDiscovered() {
      discovered = true;
    }

    public void addBacktrace(Edge left, Edge right) {
      backtraces.add(new BinaryBacktrace(left, right));
    }

    public void addBacktrace(Edge child) {
      backtraces.add(new UnaryBacktrace(child));
    }

    public void addBacktrace(String word) {
      backtraces.add(new WordBacktrace(word));
    }

    public List<Tree<String>> getTrees() {
      List<Tree<String>> trees = new ArrayList<Tree<String>>();
      for (Backtrace backtrace : backtraces) {
        if (backtrace instanceof WordBacktrace) {
          WordBacktrace wordBacktrace = (WordBacktrace) backtrace;
          trees.add(new Tree<String>(getLabel(), Collections.singletonList(new Tree<String>(wordBacktrace.getWord()))));
        } else if (backtrace instanceof UnaryBacktrace) {
          UnaryBacktrace unaryBacktrace = (UnaryBacktrace) backtrace;
          List<Tree<String>> childTrees = unaryBacktrace.getChildEdge().getTrees();
          for (Tree<String> childTree : childTrees) {
            List<Tree<String>> children = Collections.singletonList(childTree);
            trees.add(new Tree<String>(getLabel(), children));
          }
        } else if (backtrace instanceof BinaryBacktrace) {
          BinaryBacktrace binaryBacktrace = (BinaryBacktrace) backtrace;
          List<Tree<String>> leftTrees = binaryBacktrace.getLeftEdge().getTrees();
          List<Tree<String>> rightTrees = binaryBacktrace.getRightEdge().getTrees();
          for (Tree<String> leftTree : leftTrees) {
            for (Tree<String> rightTree : rightTrees) {
              List<Tree<String>> children = new ArrayList<Tree<String>>(2);
              children.add(leftTree);
              children.add(rightTree);
              trees.add(new Tree<String>(getLabel(), children));
            }
          }
        }
      }
      return trees;
    }

    public boolean equals(Object o) {
      if (this == o) return true;
      if (!(o instanceof Edge)) return false;

      final Edge edge = (Edge) o;

      if (end != edge.end) return false;
      if (start != edge.start) return false;
      if (label != null ? !label.equals(edge.label) : edge.label != null) return false;

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
      return "Edge:("+getLabel()+", "+getStart()+", "+getEnd()+")";
    }

    public Edge(String label, int start, int end) {
      this.label = label;
      this.start = start;
      this.end = end;
      this.discovered = false;
      this.backtraces = new ArrayList<Backtrace>();
    }
  }

  static class Chart {
    static class Index {
      String label;
      int pos;

      public String getLabel() {
        return label;
      }

      public int getPos() {
        return pos;
      }

      public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Index)) return false;

        final Index index = (Index) o;

        if (pos != index.pos) return false;
        if (label != null ? !label.equals(index.label) : index.label != null) return false;

        return true;
      }

      public int hashCode() {
        int result;
        result = (label != null ? label.hashCode() : 0);
        result = 29 * result + pos;
        return result;
      }

      public String toString() {
        return "Chart.Index:("+getLabel()+":"+getPos()+")";
      }

      public Index(String label, int pos) {
        this.label = label;
        this.pos = pos;
      }
    }

    Map<Index, List<Edge>> edgesByLeftIndex = new HashMap<Index, List<Edge>>();
    Map<Index, List<Edge>> edgesByRightIndex = new HashMap<Index, List<Edge>>();

    public void addEdge(Edge edge) {
      CollectionUtils.addToValueList(edgesByLeftIndex, makeLeftIndex(edge), edge);
      CollectionUtils.addToValueList(edgesByRightIndex, makeRightIndex(edge), edge);
    }

    private Index makeLeftIndex(Edge edge) {
      return new Index(edge.getLabel(), edge.getStart());
    }

    private Index makeRightIndex(Edge edge) {
      return new Index(edge.getLabel(), edge.getEnd());
    }

    public List<Edge> getEdgesByRightIndex(String label, int end) {
      return CollectionUtils.getValueList(edgesByRightIndex, new Index(label, end));
    }

    public List<Edge> getEdgesByLeftIndex(String label, int start) {
      return CollectionUtils.getValueList(edgesByLeftIndex, new Index(label, start));
    }
  }

  Lexicon lexicon;
  Grammar grammar;

  List<String> sentence;
  Chart chart;
  LinkedList<Edge> agenda;
  Map<Edge,Edge> edges;

  private void initialize() {
    agenda = new LinkedList<Edge>();
    chart = new Chart();
    edges = new HashMap<Edge,Edge>();
    for (int wordI = 0; wordI < sentence.size(); wordI++) {
      String word = (String) sentence.get(wordI);
      List<String> tagList = lexicon.getTags(word);
      if (tagList == null || tagList.isEmpty()) {
        System.err.println("Error: unknown word "+word);
        System.exit(0);
      }
      for (String tag : tagList) {
        Edge tagEdge = makeEdge(tag, wordI, wordI+1);
        if (verbose) System.err.println("Adding tagging of "+tagEdge+" for "+word);
        discoverEdge(tagEdge);
        tagEdge.addBacktrace(word);
      }
    }
    for (int wordI = 0; wordI <= sentence.size(); wordI++) {
      Edge emptyEdge = makeEdge("*e*", wordI, wordI);
      if (verbose) System.err.println("Adding empty edge of "+emptyEdge);
      discoverEdge(emptyEdge);
      emptyEdge.addBacktrace("*e*");
    }
  }

  private void processEdge(Edge edge) {
    // project unaries
    for (UnaryRule unaryRule : grammar.getUnaryRulesByChild(edge.getLabel())) {
      Edge resultEdge = makeEdge(unaryRule.getParent(), edge.getStart(), edge.getEnd());
      if (verbose) System.err.println("Using unary "+unaryRule+" on "+edge+" to create "+resultEdge);
      discoverEdge(resultEdge);
      resultEdge.addBacktrace(edge);
    }
    // project left siblings
    for (BinaryRule binaryRule : grammar.getBinaryRulesByRightChild(edge.getLabel())) {
      List<Edge> leftMatches = chart.getEdgesByRightIndex(binaryRule.getLeftChild(), edge.getStart());
      for (Edge leftMatch : leftMatches) {
        Edge resultEdge = makeEdge(binaryRule.getParent(), leftMatch.getStart(), edge.getEnd());
        if (verbose) System.err.println("Using binary "+binaryRule+" on "+leftMatch+" and "+edge+" to create "+resultEdge);
        discoverEdge(resultEdge);
        resultEdge.addBacktrace(leftMatch, edge);
      }
    }
    // project right siblings
    for (BinaryRule binaryRule : grammar.getBinaryRulesByLeftChild(edge.getLabel())) {
      List<Edge> rightMatches = chart.getEdgesByLeftIndex(binaryRule.getRightChild(), edge.getEnd());
      for (Edge rightMatch : rightMatches) {
        Edge resultEdge = makeEdge(binaryRule.getParent(), edge.getStart(), rightMatch.getEnd());
        if (verbose) System.err.println("Using binary "+binaryRule+" on "+edge+" and "+rightMatch+" to create "+resultEdge);
        discoverEdge(resultEdge);
        resultEdge.addBacktrace(edge, rightMatch);
      }
    }
  }

  private void discoverEdge(Edge edge) {
    if (edge.isDiscovered()) return;
    if (verbose) System.err.println("Discovering edge "+edge);
    edge.setDiscovered();
    agenda.addLast(edge);
  }
  public Edge makeEdge(String label, int start, int end) {
    Edge edge = new Edge(label, start, end);
    Edge canoncicalEdge = edges.get(edge);
    if (canoncicalEdge == null) {
      canoncicalEdge = edge;
      edges.put(canoncicalEdge, canoncicalEdge);
//      chart.addEdge(edge);
    }
    return canoncicalEdge;
  }

  public void parse(List<String> sentence) {
    this.sentence = sentence;
    initialize();
    while (! agenda.isEmpty()) {
      if (verbose) System.err.println("Agenda: "+agenda);
      Edge edge = agenda.removeFirst();
      if (verbose) System.err.println("Popped edge: "+edge);
      processEdge(edge);
      chart.addEdge(edge);
      if (verbose) System.err.println("Added to chart: "+edge);
    }
  }

  public List<Tree<String>> getParses() {
    Edge goalEdge = edges.get(makeEdge("ROOT", 0, sentence.size()));
    if (goalEdge == null) return Collections.emptyList();
    return goalEdge.getTrees();
  }

  public ChartParser(Lexicon lexicon, Grammar grammar) {
    this.lexicon = lexicon;
    this.grammar = grammar;
  }

  public static Tree<String> cleanTree(Tree<String> tree) {
    return Trees.spliceNodes(tree, new Filter<String>() {
      public boolean accept(String s) {
        return s.startsWith("@");
      }
    });
  }

  public static void main(String[] args) throws Exception {
    if (args.length != 3 && args.length !=4) {
      System.err.println("usage: java edu.berkeley.nlp.classical.ChartParser [-verbose] lexiconFileName grammarFileName \"sentence to parse\"");
      System.exit(0);
    }
    if (args[0].equalsIgnoreCase("-verbose") || args[0].equalsIgnoreCase("-v"))
      verbose = true;
    String lexiconFileName = args[args.length-3];
    String grammarFileName = args[args.length-2];
    String sentenceString = args[args.length-1];
    Lexicon lexicon = new Lexicon(new BufferedReader(new FileReader(lexiconFileName)));
    Grammar grammar = new Grammar(new BufferedReader(new FileReader(grammarFileName)));
    List<String> sentence = Arrays.asList(sentenceString.split("\\s+"));
    ChartParser parser = new ChartParser(lexicon, grammar);
    parser.parse(sentence);
    List<Tree<String>> parses = parser.getParses();
    for (int parseNum = 0; parseNum < parses.size(); parseNum++) {
      Tree<String> parse = (Tree<String>) parses.get(parseNum);
      parse = cleanTree(parse);
      System.out.println("PARSE "+(parseNum+1));
      System.out.print(Trees.PennTreeRenderer.render(parse));
    }
  }
}
