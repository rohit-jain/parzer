package edu.berkeley.nlp.assignments;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Scanner;
import java.util.Set;

import edu.berkeley.nlp.io.PennTreebankReader;
import edu.berkeley.nlp.ling.Tree;
import edu.berkeley.nlp.ling.Trees;
import edu.berkeley.nlp.util.BoundedList;
import edu.berkeley.nlp.util.CommandLineUtils;
import edu.berkeley.nlp.util.Counter;
import edu.berkeley.nlp.util.CounterMap;
import edu.berkeley.nlp.util.Counters;
import edu.berkeley.nlp.util.FastPriorityQueue;
import edu.berkeley.nlp.util.Interner;
import edu.berkeley.nlp.util.Pair;

/**
 * @author Dan Klein
 */
public class POSTaggerTester {

  static final String START_WORD = "START";
  static final String STOP_WORD = "STOP";
  static final String START_TAG = "<S>";
  static final String STOP_TAG = "</S>";
  static final String UNKNOWN = "<UNK>";
  static final boolean START_CAP = true;
  static final boolean STOP_CAP = true;
  static final int SUFFIX_LEN = 10;

  public boolean isCapitalised(String s){
  	return Character.isUpperCase(s.codePointAt(0));
  }

  /**
   * Tagged sentences are a bundling of a list of words and a list of their
   * tags.
   */
  static class TaggedSentence {
    List<String> words;
    List<String> tags;
    
    public int size() {
      return words.size();
    }

    public List<String> getWords() {
      return words;
    }

    public List<String> getTags() {
      return tags;
    }
    
    public String toString() {
      StringBuilder sb = new StringBuilder();
      for (int position = 0; position < words.size(); position++) {
        String word = words.get(position);
        String tag = tags.get(position);
        sb.append(word);
        sb.append("_");
        sb.append(tag);
      }
      return sb.toString();
    }

    public boolean equals(Object o) {
      if (this == o) return true;
      if (!(o instanceof TaggedSentence)) return false;

      final TaggedSentence taggedSentence = (TaggedSentence) o;

      if (tags != null ? !tags.equals(taggedSentence.tags) : taggedSentence.tags != null) return false;
      if (words != null ? !words.equals(taggedSentence.words) : taggedSentence.words != null) return false;

      return true;
    }

    public int hashCode() {
      int result;
      result = (words != null ? words.hashCode() : 0);
      result = 29 * result + (tags != null ? tags.hashCode() : 0);
      return result;
    }

    public boolean isCapitalised(String s){
    	return Character.isUpperCase(s.codePointAt(0));
    }

    public TaggedSentence(List<String> words, List<String> tags) {
      this.words = words;
      this.tags = tags;
    }
  }

  /**
   * States are pairs of tags along with a position index, representing the two
   * tags preceding that position.  So, the START state, which can be gotten by
   * State.getStartState() is [START, START, 0].  To build an arbitrary state,
   * for example [DT, NN, 2], use the static factory method
   * State.buildState("DT", "NN", 2).  There isnt' a single final state, since
   * sentences lengths vary, so State.getEndState(i) takes a parameter for the
   * length of the sentence.
   */
  static class State {

    private static transient Interner<State> stateInterner = new Interner<State>(new Interner.CanonicalFactory<State>() {
      public State build(State state) {
        return new State(state);
      }
    });

    private static transient State tempState = new State();

    public static State getStartState() {
      return buildState(START_TAG, START_TAG, 0);
    }

    public static State getStopState(int position) {
      return buildState(STOP_TAG, STOP_TAG, position);
    }

    public static State buildState(String previousPreviousTag, String previousTag, int position) {
      tempState.setState(previousPreviousTag, previousTag, position);
      return stateInterner.intern(tempState);
    }

    public static List<String> toTagList(List<State> states) {
      List<String> tags = new ArrayList<String>();
      if (states.size() > 0) {
    	try{
        tags.add(states.get(0).getPreviousPreviousTag());
        for (State state : states) {
          tags.add(state.getPreviousTag());
        }
        }
    	finally{
//    		System.out.println(states);
    	}
      }
      return tags;
    }

    public int getPosition() {
      return position;
    }

    public String getPreviousTag() {
      return previousTag;
    }

    public String getPreviousPreviousTag() {
      return previousPreviousTag;
    }

    public State getNextState(String tag) {
      return State.buildState(getPreviousTag(), tag, getPosition() + 1);
    }

    public State getPreviousState(String tag) {
      return State.buildState(tag, getPreviousPreviousTag(), getPosition() - 1);
    }

    public boolean equals(Object o) {
      if (this == o) return true;
      if (!(o instanceof State)) return false;

      final State state = (State) o;

      if (position != state.position) return false;
      if (previousPreviousTag != null ? !previousPreviousTag.equals(state.previousPreviousTag) : state.previousPreviousTag != null) return false;
      if (previousTag != null ? !previousTag.equals(state.previousTag) : state.previousTag != null) return false;

      return true;
    }

    public int hashCode() {
      int result;
      result = position;
      result = 29 * result + (previousTag != null ? previousTag.hashCode() : 0);
      result = 29 * result + (previousPreviousTag != null ? previousPreviousTag.hashCode() : 0);
      return result;
    }

    public String toString() {
      return "[" + getPreviousPreviousTag() + ", " + getPreviousTag() + ", " + getPosition() + "]";
    }

    int position;
    String previousTag;
    String previousPreviousTag;

    private void setState(String previousPreviousTag, String previousTag, int position) {
      this.previousPreviousTag = previousPreviousTag;
      this.previousTag = previousTag;
      this.position = position;
    }

    private State() {
    }

    private State(State state) {
      setState(state.getPreviousPreviousTag(), state.getPreviousTag(), state.getPosition());
    }
  }

  /**
   * A Trellis is a graph with a start state an an end state, along with
   * successor and predecessor functions.
   */
  static class Trellis <S> {
    S startState;
    S endState;
    CounterMap<S, S> forwardTransitions;
    CounterMap<S, S> backwardTransitions;

    /**
     * Get the unique start state for this trellis.
     */
    public S getStartState() {
      return startState;
    }

    public void setStartState(S startState) {
      this.startState = startState;
    }

    /**
     * Get the unique end state for this trellis.
     */
    public S getEndState() {
      return endState;
    }

    public void setStopState(S endState) {
      this.endState = endState;
    }

    /**
     * For a given state, returns a counter over what states can be next in the
     * markov process, along with the cost of that transition.  Caution: a state
     * not in the counter is illegal, and should be considered to have cost
     * Double.NEGATIVE_INFINITY, but Counters score items they don't contain as
     * 0.
     */
    public Counter<S> getForwardTransitions(S state) {
      return forwardTransitions.getCounter(state);

    }


    /**
     * For a given state, returns a counter over what states can precede it in
     * the markov process, along with the cost of that transition.
     */
    public Counter<S> getBackwardTransitions(S state) {
      return backwardTransitions.getCounter(state);
    }

    public void setTransitionCount(S start, S end, double count) {
      forwardTransitions.setCount(start, end, count);
      backwardTransitions.setCount(end, start, count);
    }

    public Trellis() {
      forwardTransitions = new CounterMap<S, S>();
      backwardTransitions = new CounterMap<S, S>();
    }
  }

  /**
   * A TrellisDecoder takes a Trellis and returns a path through that trellis in
   * which the first item is trellis.getStartState(), the last is
   * trellis.getEndState(), and each pair of states is conntected in the
   * trellis.
   */
  static interface TrellisDecoder <S> {
    List<S> getBestPath(Trellis<S> trellis, int sentenceLength);
  }
  
  
  
  static class ViterbiDecoder <S> implements TrellisDecoder<S> {
	    public List<S> getBestPath(Trellis<S> trellis, int sentenceLength) {
	      List<S> states = new ArrayList<S>();
	      Set<S> allStates = new HashSet<S>();
	      List<Map<S,S>> backPointers = new ArrayList<Map<S,S>>();
	      
	      CounterMap< Integer , S> pie = new CounterMap< Integer , S>();
	      
	      S currentState = trellis.getStartState();
	      
	      // set position to 0
	      int position = 0;
	      allStates.add(currentState);
	      allStates.add(trellis.getEndState());
	      backPointers.add(position, null);
	      
	      position += 1; 
	      while( position < sentenceLength+3 ){
	    	  Set<S> allTempStates = new HashSet<S>();
	    	  for (S s: allStates){
	    		  allTempStates.addAll(trellis.getForwardTransitions(s).keySet());
	    	  }
	    	  allStates.addAll(allTempStates);
	    	  position += 1;
	      }
	      
	      position = 0;
	      // set the start state probability to 1 for position 0
	      pie.setCount(position, currentState, 0.0);
	      
	      position += 1;
	      
	      while( position < sentenceLength+3 ){
	        Map<S,S> stateBackPointers = new HashMap<S,S>();
	        for(S s: allStates){
	        	Counter<S> viterbiTransitions = new Counter<S>();
	  	      
	        	for(S s_i: allStates){
	        		if (pie.getCounter(position - 1).keySet().contains(s_i)){
	        			double oldPie = pie.getMinCount(position-1 , s_i);
	        			double pq;
	        			if (trellis.getForwardTransitions(s_i).keySet().contains(s)){
		        			pq = trellis.getForwardTransitions(s_i).getCount(s);
			        		double p = oldPie + pq ;
				        	viterbiTransitions.setCount(s_i, p);
		        		}
	        			else{
	        				viterbiTransitions.setCount(s_i, Double.NEGATIVE_INFINITY);
	        			}
	        		}
        			else{
        				viterbiTransitions.setCount(s_i, Double.NEGATIVE_INFINITY);
        			}
	        	}
	        	S maxState = viterbiTransitions.argMax();
	        	stateBackPointers.put(s, maxState);
	        	pie.setCount(position, s, viterbiTransitions.getCount(maxState));
	        }
	        backPointers.add(position, stateBackPointers);
	        position += 1;
	      }
	      
	      S lastBestState = trellis.getEndState();
	      states.add(lastBestState);
	      position = sentenceLength+2;
	      
	      while(position > 0 ){
	    	  
	    	  Map<S,S> lastStateBackpointers = backPointers.get(position);
	    	  S newBestState = lastStateBackpointers.get(lastBestState);
	    	  states.add(newBestState);
	    	  lastBestState = newBestState;
	    	  position -= 1;
	    	  
	      }


	      List<S> tagSeq = new ArrayList<S>();
	      position = sentenceLength+2;
	      
	      while(states.size()>0){
	    	  tagSeq.add(states.remove(position));
	    	  position -= 1;
	      }
	      return tagSeq;
	    }
	  }

  static class GreedyDecoder <S> implements TrellisDecoder<S> {
	    public List<S> getBestPath(Trellis<S> trellis, int sentenceLength) {
	      List<S> states = new ArrayList<S>();
	      S currentState = trellis.getStartState();
	      states.add(currentState);
	      while (!currentState.equals(trellis.getEndState())) {
	        Counter<S> transitions = trellis.getForwardTransitions(currentState);
	        S nextState = transitions.argMax();
	        states.add(nextState);
	        currentState = nextState;
	      }
	      return states;
	    }
	  }

  static class POSTagger {

    LocalTrigramScorer localTrigramScorer;
    TrellisDecoder<State> trellisDecoder;

    // chop up the training instances into local contexts and pass them on to the local scorer.
    public void train(List<TaggedSentence> taggedSentences) {
      localTrigramScorer.train(extractLabeledLocalTrigramContexts(taggedSentences));
    }

    // chop up the validation instances into local contexts and pass them on to the local scorer.
    public void validate(List<TaggedSentence> taggedSentences) {
      localTrigramScorer.validate(extractLabeledLocalTrigramContexts(taggedSentences));
    }

    private List<LabeledLocalTrigramContext> extractLabeledLocalTrigramContexts(List<TaggedSentence> taggedSentences) {
      List<LabeledLocalTrigramContext> localTrigramContexts = new ArrayList<LabeledLocalTrigramContext>();
      for (TaggedSentence taggedSentence : taggedSentences) {
        localTrigramContexts.addAll(extractLabeledLocalTrigramContexts(taggedSentence));
      }
      return localTrigramContexts;
    }

    private List<LabeledLocalTrigramContext> extractLabeledLocalTrigramContexts(TaggedSentence taggedSentence) {
      List<LabeledLocalTrigramContext> labeledLocalTrigramContexts = new ArrayList<LabeledLocalTrigramContext>();
      List<String> words = new BoundedList<String>(taggedSentence.getWords(), START_WORD, STOP_WORD);
      List<String> tags = new BoundedList<String>(taggedSentence.getTags(), START_TAG, STOP_TAG);
      for (int position = 0; position <= taggedSentence.size() + 1; position++) {
        labeledLocalTrigramContexts.add(new LabeledLocalTrigramContext(words, position, tags.get(position - 2), tags.get(position - 1), tags.get(position)));
      }
      return labeledLocalTrigramContexts;
    }

    /**
     * Builds a Trellis over a sentence, by starting at the state State, and
     * advancing through all legal extensions of each state already in the
     * trellis.  You should not have to modify this code (or even read it,
     * really).
     */
    private Trellis<State> buildTrellis(List<String> sentence) {
      Trellis<State> trellis = new Trellis<State>();
      trellis.setStartState(State.getStartState());
      State stopState = State.getStopState(sentence.size() + 2);
      trellis.setStopState(stopState);
      Set<State> states = Collections.singleton(State.getStartState());
      for (int position = 0; position <= sentence.size() + 1; position++) {
        Set<State> nextStates = new HashSet<State>();
        for (State state : states) {
          if (state.equals(stopState))
            continue;
          LocalTrigramContext localTrigramContext = new LocalTrigramContext(sentence, position, state.getPreviousPreviousTag(), state.getPreviousTag());
          Counter<String> tagScores = localTrigramScorer.getLogScoreCounter(localTrigramContext);
          for (String tag : tagScores.keySet()) {
            double score = tagScores.getCount(tag);
            State nextState = state.getNextState(tag);
            trellis.setTransitionCount(state, nextState, score);
            nextStates.add(nextState);
          }
        }
//        System.out.println("States: "+nextStates);
        states = nextStates;
      }
      return trellis;
    }

    // to tag a sentence: build its trellis and find a path through that trellis
    public List<String> tag(List<String> sentence) {
      Trellis<State> trellis = buildTrellis(sentence);
//      System.out.println(sentence);
//      System.out.println(sentence.size());
      List<State> states = trellisDecoder.getBestPath(trellis, sentence.size());
      List<String> tags = State.toTagList(states);
//      System.out.println(tags);
      tags = stripBoundaryTags(tags);
      return tags;
    }

    /**
     * Scores a tagging for a sentence.  Note that a tag sequence not accepted
     * by the markov process should receive a log score of
     * Double.NEGATIVE_INFINITY.
     */
    public double scoreTagging(TaggedSentence taggedSentence) {
      double logScore = 0.0;
      List<LabeledLocalTrigramContext> labeledLocalTrigramContexts = extractLabeledLocalTrigramContexts(taggedSentence);
      for (LabeledLocalTrigramContext labeledLocalTrigramContext : labeledLocalTrigramContexts) {
        Counter<String> logScoreCounter = localTrigramScorer.getLogScoreCounter(labeledLocalTrigramContext);
        String currentTag = labeledLocalTrigramContext.getCurrentTag();
        if (logScoreCounter.containsKey(currentTag)) {
          logScore += logScoreCounter.getCount(currentTag);
        } else {
          logScore += Double.NEGATIVE_INFINITY;
        }
      }
      return logScore;
    }

    private List<String> stripBoundaryTags(List<String> tags) {
      return tags.subList(2, tags.size() - 2);
    }

    public POSTagger(LocalTrigramScorer localTrigramScorer, TrellisDecoder<State> trellisDecoder) {
      this.localTrigramScorer = localTrigramScorer;
      this.trellisDecoder = trellisDecoder;
    }
  }

  /**
   * A LocalTrigramContext is a position in a sentence, along with the previous
   * two tags -- basically a FeatureVector.
   */
  static class LocalTrigramContext {
    List<String> words;
    int position;
    String previousTag;
    String previousPreviousTag;
    Boolean previousCaps;
    Boolean previousPreviousCaps;
    
    public boolean isCapitalised(String s){
    	return Character.isUpperCase(s.codePointAt(0));
    }
    
    public List<String> getWords() {
      return words;
    }
    
    public String getCurrentWord() {
      return words.get(position);
    }

    public int getPosition() {
      return position;
    }

    public String getPreviousTag() {
      return previousTag;
    }

    public String getPreviousPreviousTag() {
      return previousPreviousTag;
    }
    
    public Boolean getPreviousCaps() {
        return previousCaps;
    }

	public Boolean getPreviousPreviousCaps() {
	  return previousPreviousCaps;
	}

    public String toString() {
      return "[" + getPreviousPreviousTag() + ", " + getPreviousTag() + ", " + getCurrentWord() + "]";
    }

    public LocalTrigramContext(List<String> words, int position, String previousPreviousTag, String previousTag) {
      this.words = words;
      this.position = position;
      this.previousTag = previousTag;
      this.previousPreviousTag = previousPreviousTag;
      this.previousCaps = isCapitalised(words.get(position-1));
      this.previousPreviousCaps = isCapitalised(words.get(position-2));
    }
  }

  /**
   * A OldLabeledLocalTrigramContext is a context plus the correct tag for that
   * position -- basically a LabeledFeatureVector
   */
  static class OldLabeledLocalTrigramContext extends LocalTrigramContext {
    String currentTag;

    public String getCurrentTag() {
      return currentTag;
    }

    public String toString() {
      return "[" + getPreviousPreviousTag() + ", " + getPreviousTag() + ", " + getCurrentWord() + "_" + getCurrentTag() + "]";
    }

    public OldLabeledLocalTrigramContext(List<String> words, int position, String previousPreviousTag, String previousTag, String currentTag) {
      super(words, position, previousPreviousTag, previousTag);
      this.currentTag = currentTag;
    }
  }
  
  /**
   * A LabeledLocalTrigramContext is a context plus the correct tag for that
   * position -- basically a LabeledFeatureVector
   */
  static class LabeledLocalTrigramContext extends LocalTrigramContext {
    String currentTag;
    Boolean currentCaps;

    public String getCurrentTag() {
      return currentTag;
    }

    public Boolean getCurrentCaps(){
    	return currentCaps;
    }
    
    public String toString() {
      return "[" + getPreviousPreviousTag() + ", " + getPreviousTag() + ", " + getCurrentWord() + "_" + getCurrentTag() + "]";
    }

    public LabeledLocalTrigramContext(List<String> words, int position, String previousPreviousTag, String previousTag, String currentTag) {
      super(words, position, previousPreviousTag, previousTag);
      this.currentTag = currentTag;
      this.currentCaps = isCapitalised(words.get(position));
    }
  }


  /**
   * LocalTrigramScorers assign scores to tags occuring in specific
   * LocalTrigramContexts.
   */
  static interface LocalTrigramScorer {
    /**
     * The Counter returned should contain log probabilities, meaning if all
     * values are exponentiated and summed, they should sum to one (if it's a 
	 * single conditional pobability). For efficiency, the Counter can 
	 * contain only the tags which occur in the given context 
	 * with non-zero model probability.
     */
    Counter<String> getLogScoreCounter(LocalTrigramContext localTrigramContext);

    void train(List<LabeledLocalTrigramContext> localTrigramContexts);

    void validate(List<LabeledLocalTrigramContext> localTrigramContexts);
  }


  static class HMMTagScorer implements LocalTrigramScorer {

    boolean restrictTrigrams; // if true, assign log score of Double.NEGATIVE_INFINITY to illegal tag trigrams.

    CounterMap<String, Pair<String, Boolean>> tagsToTags = new CounterMap<String, Pair<String, Boolean>>();
    CounterMap<String, Pair<String, Boolean>> previousTagToTags = new CounterMap<String, Pair<String, Boolean>>();
    CounterMap<Pair<String, Boolean>, String> tagsToWords = new CounterMap<Pair<String, Boolean>, String>();
    
    CounterMap<String, Pair<String, Boolean>> suffixToTags = new CounterMap<String, Pair<String, Boolean>>();
    CounterMap<String, Pair<String, Boolean>> capSuffixToTags = new CounterMap<String, Pair<String, Boolean>>();

    CounterMap<Pair<String, Boolean>, String> tagsToSuffix = new CounterMap<Pair<String, Boolean>, String>();
    CounterMap<Pair<String, Boolean>, String> capTagsToSuffix = new CounterMap<Pair<String, Boolean>, String>();

    CounterMap<String, Pair<String, Boolean>> smoothedSuffixToTags = new CounterMap<String, Pair<String, Boolean>>();
    CounterMap<String, Pair<String, Boolean>> capSmoothedSuffixToTags = new CounterMap<String, Pair<String, Boolean>>();
    
    CounterMap<String, Pair<String, Boolean>> wordToTags = new CounterMap<String, Pair<String, Boolean>>();
    
    Counter<String> knownSuffixes = new Counter<String>();
    Counter<String> capKnownSuffixes = new Counter<String>();
    
    Counter<String> knownWords = new Counter<String>();
    Set<String> seenTagTrigrams = new HashSet<String>();
    
    CounterMap<Boolean, String> infrequentTags = new CounterMap<Boolean, String>();
    Counter<Pair<String, Boolean>> t3 = new Counter<Pair<String, Boolean>>();
    Counter<Pair<String, Boolean>> t2 = new Counter<Pair<String, Boolean>>();
    Counter<String> t1t2 = new Counter<String>();
    Counter<String> t2t3 = new Counter<String>();
    Counter<String> t1t2t3 = new Counter<String>();

    Double lambda1 = 0.0;
    Double lambda2 = 0.0;
    Double lambda3 = 0.0;

    public String getSuffix(String word, int length){
    	int len = Math.min(word.length(), length);
    	return word.substring(word.length() - len);
    }
    
    public int getHistorySize() {
      return 2;
    }
    
    public boolean isCapitalised(String s){
      	return Character.isUpperCase(s.codePointAt(0));
    }
    

    public Double smoothTransitionProbability(CounterMap<String, Pair<String, Boolean>> trigramCounter, Pair<String, Boolean> tagPair, Pair<String, Boolean> previousTagPair, String precedingTags, String bi_tag){
    	Double p = 0.0;
    	Double p_t3 = t3.getCount(tagPair)/t3.totalCount();
    	Double p_t3t2 = t2t3.getCount(bi_tag)/t2.getCount(previousTagPair);
    	Double p_t1t2t3 = trigramCounter.getCount(precedingTags, tagPair);
    	p = (lambda1 * p_t3) + (lambda2 * p_t3t2) + (lambda3 * p_t1t2t3);
    	return p;
    }
    
    public Counter<String> getLogScoreCounter(LocalTrigramContext localTrigramContext) {
        int position = localTrigramContext.getPosition();
        String word = localTrigramContext.getWords().get(position);
        String previousTag = localTrigramContext.getPreviousTag();
        String previousPreviousTag = localTrigramContext.getPreviousPreviousTag();
        Boolean previousCaps = localTrigramContext.getPreviousCaps();
        Boolean previousPreviousCaps = localTrigramContext.getPreviousPreviousCaps();
        String precedingTags = makeBigramString(previousPreviousTag + previousPreviousCaps, previousTag + previousCaps);
        Pair<String, Boolean> previousTagPair = new Pair<String, Boolean>(previousTag, previousCaps);
        Counter<String> logScoreCounter = new Counter<String>();
        
        
        if (word == STOP_WORD){
          Pair<String, Boolean> tagPair = new Pair<String, Boolean>(STOP_TAG, true);
          String previousCurrent = makeBigramString(previousTag + previousCaps, STOP_TAG + true);          
          
    	  double transition_probability = smoothTransitionProbability(tagsToTags, tagPair, previousTagPair, precedingTags, previousCurrent);
    	  if (transition_probability == 0){
    		  transition_probability  = Double.NEGATIVE_INFINITY;
    	  }
    	  else{
    		  transition_probability = Math.log(transition_probability);
    	  }
    	  logScoreCounter.setCount(STOP_TAG, transition_probability);
    	  return logScoreCounter;
        }
        
        if (knownWords.keySet().contains(word)){
        	Set<Pair<String, Boolean>> candidateTags = wordToTags.getCounter(word).keySet();
        	for (Pair<String, Boolean> candidateTagPair : candidateTags) {
              String previousCurrent = makeBigramString(previousTag + previousCaps, candidateTagPair.getFirst() + candidateTagPair.getSecond());
        	  String candidateTag = candidateTagPair.getFirst();
        	  
        	  double transition_probability = smoothTransitionProbability(tagsToTags, candidateTagPair, previousTagPair, precedingTags, previousCurrent);
        	  double emission_probability = tagsToWords.getCount(candidateTagPair,word);
        	  
          	  if ((transition_probability == 0) || (emission_probability == 0)){
          		  logScoreCounter.setCount(candidateTag, Double.NEGATIVE_INFINITY);
        	  }
        	  else{
        		  transition_probability = Math.log(transition_probability);
        		  emission_probability = Math.log(emission_probability);
			      double logScore = (transition_probability) + (emission_probability);
			      logScoreCounter.setCount(candidateTag, (logScore));
        	  }
          	}
        }
        else{
        	// look at suffixes

        	String bestSuffix = getSuffix(word, SUFFIX_LEN);

    		
        	// get candidate tags
    		Set<Pair<String,Boolean>> candidateTags;
    		CounterMap<Pair<String,Boolean>,String> tagsSuffix;
    		
    		if(!isCapitalised(word)){
            	for(int i=SUFFIX_LEN; i > 0; i--){
            		String newSuffix = getSuffix(word, i);
            		if (knownSuffixes.containsKey(newSuffix)){
            			bestSuffix = newSuffix;
            			break;
            		}
            	}

    			candidateTags = smoothedSuffixToTags.getCounter(bestSuffix).keySet();
    			tagsSuffix = tagsToSuffix;
    		}
    		else{
            	for(int i=SUFFIX_LEN; i > 0; i--){
            		String newSuffix = getSuffix(word, i);
            		if (capKnownSuffixes.containsKey(newSuffix)){
            			bestSuffix = newSuffix;
            			break;
            		}
            	}

    			candidateTags = capSmoothedSuffixToTags.getCounter(bestSuffix).keySet();
    			tagsSuffix = capTagsToSuffix;
    		}
    		
    		
    		// get probability for each candidate tag
    		for (Pair<String,Boolean> candidateTagPair : candidateTags) {
              String previousCurrent = makeBigramString(previousTag + previousCaps, candidateTagPair.getFirst() + candidateTagPair.getSecond());
          	  String candidateTag = candidateTagPair.getFirst();

          	  double transition_probability = smoothTransitionProbability(tagsToTags, candidateTagPair, previousTagPair, precedingTags, previousCurrent);
          	  double emission_probability = tagsSuffix.getCount(candidateTagPair, bestSuffix);// * tagsToWords.getCount(candidateTag, UNKNOWN);

          	  if ((transition_probability == 0) || (emission_probability == 0)){
          		  logScoreCounter.setCount(candidateTag, Double.NEGATIVE_INFINITY);
        	  }
        	  else{
        		  transition_probability = Math.log(transition_probability);
        		  emission_probability = Math.log(emission_probability);
			      double logScore = (transition_probability) + (emission_probability);
			      logScoreCounter.setCount(candidateTag, (logScore));
        	  }
            }
        }        
        
        return logScoreCounter;
      }

        
    private String makeBigramString(String a, String b){
        return a + " " + b;
    }
    
    private String makeTrigramString(String previousPreviousTag, String previousTag, String currentTag) {
      return previousPreviousTag + " " + previousTag + " " + currentTag;
    }
    
    public Double pHat(int l, String s, Pair<String, Boolean> t, double theta){
    	if(l == 0){
    		// t3 is basically known tags
    		return infrequentTags.getCounter(false).getCount(t.getFirst());
    	}
    	return (suffixToTags.getCounter(getSuffix(s, l)).getCount(t) + (theta * pHat(l-1, s, t, theta)) )/( 1 + theta);
    }
    
    public Double pHatCap(int l, String s, Pair<String, Boolean> t, double theta){
    	if(l == 0){
    		// t3 is basically known tags
    		return infrequentTags.getCounter(true).getCount(t.getFirst());
    	}
    	return (capSuffixToTags.getCounter(getSuffix(s, l)).getCount(t) + (theta * pHatCap(l-1, s, t, theta)) )/( 1 + theta);
    }
    

    public void train(List<LabeledLocalTrigramContext> labeledLocalTrigramContexts) {
      
      // collect word-tag counts
      for (LabeledLocalTrigramContext labeledLocalTrigramContext : labeledLocalTrigramContexts) {
        String word = labeledLocalTrigramContext.getCurrentWord();
        String tag = labeledLocalTrigramContext.getCurrentTag();
        Boolean caps = labeledLocalTrigramContext.getCurrentCaps();
        String previousTag = labeledLocalTrigramContext.getPreviousTag();
        String previousPreviousTag = labeledLocalTrigramContext.getPreviousPreviousTag();
        Boolean previousCaps = labeledLocalTrigramContext.getPreviousCaps();
        Boolean previousPreviousCaps = labeledLocalTrigramContext.getPreviousPreviousCaps();
        String precedingTags = makeBigramString(previousPreviousTag + previousPreviousCaps, previousTag + previousCaps);
        String trigram = makeTrigramString(previousPreviousTag + previousPreviousCaps, previousTag + previousCaps, tag + caps);        
        Pair<String, Boolean> tagPair = new Pair<String, Boolean>(tag,caps);
        Pair<String, Boolean> previousTagPair = new Pair<String, Boolean>(previousTag, previousCaps);
        
        tagsToTags.incrementCount(precedingTags, tagPair , 1.0);
        tagsToWords.incrementCount(tagPair, word, 1.0);
        wordToTags.incrementCount(word, tagPair, 1.0);
        previousTagToTags.incrementCount(previousTag + previousCaps, tagPair, 1.0);
        seenTagTrigrams.add(trigram);
        
        knownWords.incrementCount(word, 1.0);
        
        t2.incrementCount(previousTagPair, 1.0);
        t3.incrementCount(tagPair,1.0);
        t1t2.incrementCount(precedingTags, 1.0);
        t2t3.incrementCount(makeBigramString(previousTag+previousCaps, tag+caps), 1.0);
        t1t2t3.incrementCount(trigram, 1.0);
        
      }
      
      for (LabeledLocalTrigramContext labeledLocalTrigramContext : labeledLocalTrigramContexts) {
          String word = labeledLocalTrigramContext.getCurrentWord();
          String tag = labeledLocalTrigramContext.getCurrentTag();
          Boolean caps = labeledLocalTrigramContext.getCurrentCaps();
          Pair<String, Boolean> tagPair = new Pair<String, Boolean>(tag, caps);
          if(knownWords.getCount(word) < 11){
        	  
        	  if(!isCapitalised(word)){
		          for( int i=1; i <= Math.min(SUFFIX_LEN, word.length()); i++){
		  			String suffix = getSuffix(word, i);
		  			suffixToTags.incrementCount(suffix, tagPair, 1.0);
		  			knownSuffixes.incrementCount(suffix, 1.0);
		          }
        	  }
        	  else{
		          for( int i=1; i <= Math.min(SUFFIX_LEN, word.length()); i++){
		  			String suffix = getSuffix(word, i);
		  			capSuffixToTags.incrementCount(suffix, tagPair, 1.0);
		  			capKnownSuffixes.incrementCount(suffix, 1.0);
		          }
        	  }
          }
          infrequentTags.incrementCount( caps, tag, 1.0);
//          }
       }
      
      // TODO: add one unknown to every tag
      for (Pair<String, Boolean> tag: tagsToWords.keySet()){
    	  tagsToWords.incrementCount(tag, UNKNOWN, 1.0);
      }

      tagsToTags = Counters.conditionalNormalize(tagsToTags);
      tagsToWords = Counters.conditionalNormalize(tagsToWords);
      previousTagToTags = Counters.conditionalNormalize(previousTagToTags);
      suffixToTags = Counters.conditionalNormalize(suffixToTags);
      capSuffixToTags = Counters.conditionalNormalize(capSuffixToTags);
      
      knownSuffixes = Counters.normalize(knownSuffixes);
      capKnownSuffixes = Counters.normalize(capKnownSuffixes);

      infrequentTags = Counters.conditionalNormalize(infrequentTags);
      
      Double theta = infrequentTags.getCounter(false).standardDeviation();
      Double capTheta = infrequentTags.getCounter(true).standardDeviation();
      System.out.printf("Theta : %f\n",theta);
      System.out.printf("Theta : %f\n",capTheta);
      // smoothed suffix probabilities
      for (String s: suffixToTags.keySet()){
    	for (Pair<String, Boolean> t: suffixToTags.getCounter(s).keySet()){
    		// different distribution to protect old calc
    		smoothedSuffixToTags.setCount(s, t, pHat(s.length(), s, t, theta));
    	}  
      }
      
      for (String s: capSuffixToTags.keySet()){
      	for (Pair<String, Boolean> t: capSuffixToTags.getCounter(s).keySet()){
      		// different distribution to protect old calc
      		capSmoothedSuffixToTags.setCount(s, t, pHatCap(s.length(), s, t, theta));
      	}  
      }
      
      // apply bayes rule to get tags to suffix
      for(String s: smoothedSuffixToTags.keySet()){
    	  for(Pair<String, Boolean> t: smoothedSuffixToTags.getCounter(s).keySet()){
    		  Double p = (smoothedSuffixToTags.getCounter(s).getCount(t) * knownSuffixes.getCount(s))/infrequentTags.getCounter(false).getCount(t.getFirst());
    		  tagsToSuffix.incrementCount(t, s, p);
    	  }
      }
      
   // apply bayes rule to get tags to suffix
      for(String s: capSmoothedSuffixToTags.keySet()){
    	  for(Pair<String, Boolean> t: capSmoothedSuffixToTags.getCounter(s).keySet()){
    		  Double p = (capSmoothedSuffixToTags.getCounter(s).getCount(t) * capKnownSuffixes.getCount(s))/infrequentTags.getCounter(true).getCount(t.getFirst());
    		  capTagsToSuffix.incrementCount(t, s, p);
    	  }
      }
    }

    public void validate(List<LabeledLocalTrigramContext> labeledLocalTrigramContexts) {
      // tune using linear interpolation
      for (LabeledLocalTrigramContext labeledLocalTrigramContext : labeledLocalTrigramContexts) {
          String tag = labeledLocalTrigramContext.getCurrentTag();
          Boolean caps = labeledLocalTrigramContext.getCurrentCaps();
          String previousTag = labeledLocalTrigramContext.getPreviousTag();
          String previousPreviousTag = labeledLocalTrigramContext.getPreviousPreviousTag();
          Boolean previousCaps = labeledLocalTrigramContext.getPreviousCaps();
          Boolean previousPreviousCaps = labeledLocalTrigramContext.getPreviousPreviousCaps();
          String precedingTags = makeBigramString(previousPreviousTag + previousPreviousCaps, previousTag + previousCaps);
          String trigram = makeTrigramString(previousPreviousTag + previousPreviousCaps, previousTag + previousCaps, tag + caps);        
          Pair<String, Boolean> tagPair = new Pair<String, Boolean>(tag,caps);
          Pair<String, Boolean> previousTagPair = new Pair<String, Boolean>(previousTag, previousCaps);

          
          Double fif2f3 = t1t2t3.getCount(trigram);
          Double f1f2 = t1t2.getCount(precedingTags);
          Double f2f3 = t2t3.getCount(makeBigramString(previousTag + previousCaps, tag + caps));
          Double f2 = t2.getCount(previousTagPair);
          Double f3 = t3.getCount(tagPair);
          Double N = t3.totalCount();
          if(fif2f3>0){
        	  Double v3 = (fif2f3 - 1)/(f1f2 -1);
        	  Double v2 = (f2f3 -1)/(f2 -1);
        	  Double v1 = (f3 -1)/(N-1);
        	  
        	  if ((v3>v2) && (v3>v1)){
        		  lambda3 += fif2f3;
        	  }
        	  else if((v2>v1) && (v2>v3)){
        		  lambda2 += fif2f3;
        	  }
        	  else if((v1>v2) && (v1>v3)){
        		  lambda1 += fif2f3;
        	  }
        	  else if ((v3==v2) && (v3>v1)){
        		  lambda3 += (fif2f3/2);
        		  lambda2 += (fif2f3/2);
        	  }
        	  else if ((v2==v1) && (v2>v3)){
        		  lambda1 += (fif2f3/2);
        		  lambda2 += (fif2f3/2);
        	  }
        	  else if ((v1==v3) && (v1>v2)){
        		  lambda3 += (fif2f3/2);
        		  lambda1 += (fif2f3/2);
        	  }
        	  else if ((v1==v3) && (v1==v2)){
        		  lambda1 += (fif2f3/3);
        		  lambda2 += (fif2f3/3);
        		  lambda3 += (fif2f3/3);
        	  }

          }
          
      }
      
      Double sum = lambda1 + lambda2 + lambda3;
      lambda1 = lambda1/sum;
      lambda2 = lambda2/sum;
      lambda3 = lambda3/sum;
      System.out.println(lambda1);
      System.out.println(lambda2);
      System.out.println(lambda3);
    }

    public HMMTagScorer(boolean restrictTrigrams) {
      this.restrictTrigrams = restrictTrigrams;
    }
  }

  
  private static List<TaggedSentence> readTaggedSentences(String path, int low, int high) {
    Collection<Tree<String>> trees = PennTreebankReader.readTrees(path, low, high);
    List<TaggedSentence> taggedSentences = new ArrayList<TaggedSentence>();
    Trees.TreeTransformer<String> treeTransformer = new Trees.EmptyNodeStripper();
    for (Tree<String> tree : trees) {
      tree = treeTransformer.transformTree(tree);
      List<String> words = new BoundedList<String>(new ArrayList<String>(tree.getYield()), START_WORD, STOP_WORD);
      List<String> tags = new BoundedList<String>(new ArrayList<String>(tree.getPreTerminalYield()), START_TAG, STOP_TAG);
      taggedSentences.add(new TaggedSentence(words, tags));
    }
    return taggedSentences;
  }
  
  private static List<TaggedSentence> readTaggedSentencesTwitter(String path, int low, int high) {
	    List<TaggedSentence> taggedSentences = new ArrayList<TaggedSentence>();
	    int  lineNumber = 0;
	    try {
	        FileReader fr = new FileReader(path);
	        Scanner in = new Scanner(fr);
	        while (in.hasNext()) {
	        	lineNumber++;
	        	String temp = in.nextLine();
	        	List<String> tokens = new ArrayList<String>();
	        	List<String> posTags = new ArrayList<String>();
	        	while(temp.length()>0 && in.hasNext()){
	        		String[] line = temp.split(" ");
	        		tokens.add(line[0]);
	        		posTags.add(line[1]);
	        		temp = in.nextLine();
	        	}
	        	if ((lineNumber >=low) && (lineNumber <= high)){
		        	List<String> words = new BoundedList<String>(tokens, START_WORD, STOP_WORD);
		            List<String> tags = new BoundedList<String>(posTags, START_TAG, STOP_TAG);
		            taggedSentences.add(new TaggedSentence(words, tags));
	        	}
	        }
	    }
	    catch (FileNotFoundException ex) {
	    	System.out.println("Tweets file not found");
	    }
	    return taggedSentences;
  }

  private static void evaluateTagger(POSTagger posTagger, List<TaggedSentence> taggedSentences, Set<String> trainingVocabulary, boolean verbose) {
    double numTags = 0.0;
    double numTagsCorrect = 0.0;
    double numUnknownWords = 0.0;
    double numUnknownWordsCorrect = 0.0;
    int numDecodingInversions = 0;
    CounterMap<String, String> errors = new CounterMap<String, String>();
    Counter<String> goldError = new Counter<String>();
    
    for (TaggedSentence taggedSentence : taggedSentences) {
      List<String> words = taggedSentence.getWords();
      List<String> goldTags = taggedSentence.getTags();
      List<String> guessedTags = posTagger.tag(words);
      for (int position = 0; position < words.size() - 1; position++) {
        String word = words.get(position);
        String goldTag = goldTags.get(position);
        String guessedTag = guessedTags.get(position);
        if (guessedTag.equals(goldTag)){
          numTagsCorrect += 1.0;
        }
        else{
        	errors.incrementCount(goldTag, guessedTag, 1.0);
        	goldError.incrementCount(goldTag, 1.0);
        }
        numTags += 1.0;
        if (!trainingVocabulary.contains(word)) {
          if (guessedTag.equals(goldTag))
            numUnknownWordsCorrect += 1.0;
          numUnknownWords += 1.0;
        }
      }
      double scoreOfGoldTagging = posTagger.scoreTagging(taggedSentence);
      double scoreOfGuessedTagging = posTagger.scoreTagging(new TaggedSentence(words, guessedTags));
      if (scoreOfGoldTagging > scoreOfGuessedTagging) {
        numDecodingInversions++;
        if (verbose) System.out.println("WARNING: Decoder suboptimality detected.  Gold tagging has higher score than guessed tagging.");
      }
      if (verbose) System.out.println(alignedTaggings(words, goldTags, guessedTags, true) + "\n");
    }
    
    Map<String, Double> topErrors = goldError.getTopK(5);
    Set<String> tags = new HashSet<String>();
    
    System.out.println(topErrors);
    for (String t: topErrors.keySet()){
    	tags.addAll(errors.getCounter(t).getTopK(5).keySet());
    }
    
    System.out.print("Tags ");
    for (String v: tags){
    	System.out.print(v + " , ");
    }
    
    System.out.print("\n");
    
    for (String k: topErrors.keySet()){
    	System.out.print(k + " : ");
    	for(String v: tags){
    		System.out.print( errors.getCount(k,v) + " , " );
    	}
    	System.out.print("\n");
    }
    
    System.out.println("Tag Accuracy: " + (numTagsCorrect / numTags) + " (Unknown Accuracy: " + (numUnknownWordsCorrect / numUnknownWords) + ")  Decoder Suboptimalities Detected: " + numDecodingInversions);
  }

  // pretty-print a pair of taggings for a sentence, possibly suppressing the tags which correctly match
  private static String alignedTaggings(List<String> words, List<String> goldTags, List<String> guessedTags, boolean suppressCorrectTags) {
    StringBuilder goldSB = new StringBuilder("Gold Tags: ");
    StringBuilder guessedSB = new StringBuilder("Guessed Tags: ");
    StringBuilder wordSB = new StringBuilder("Words: ");
    for (int position = 0; position < words.size(); position++) {
      equalizeLengths(wordSB, goldSB, guessedSB);
      String word = words.get(position);
      String gold = goldTags.get(position);
      String guessed = guessedTags.get(position);
      wordSB.append(word);
      if (position < words.size() - 1)
        wordSB.append(' ');
      boolean correct = (gold.equals(guessed));
      if (correct && suppressCorrectTags)
        continue;
      guessedSB.append(guessed);
      goldSB.append(gold);
    }
    return goldSB + "\n" + guessedSB + "\n" + wordSB;
  }

  private static void equalizeLengths(StringBuilder sb1, StringBuilder sb2, StringBuilder sb3) {
    int maxLength = sb1.length();
    maxLength = Math.max(maxLength, sb2.length());
    maxLength = Math.max(maxLength, sb3.length());
    ensureLength(sb1, maxLength);
    ensureLength(sb2, maxLength);
    ensureLength(sb3, maxLength);
  }

  private static void ensureLength(StringBuilder sb, int length) {
    while (sb.length() < length) {
      sb.append(' ');
    }
  }

  private static Set<String> extractVocabulary(List<TaggedSentence> taggedSentences) {
    Set<String> vocabulary = new HashSet<String>();
    for (TaggedSentence taggedSentence : taggedSentences) {
      List<String> words = taggedSentence.getWords();
      vocabulary.addAll(words);
    }
    return vocabulary;
  }

  public static void main(String[] args) {
    // Parse command line flags and arguments
    Map<String, String> argMap = CommandLineUtils.simpleCommandLineParser(args);

    // Set up default parameters and settings
    String basePath = ".";
    String tweetPath = ".";
    boolean verbose = false;
    boolean useValidation = true;

    // Update defaults using command line specifications

    // The path to the assignment data
    if (argMap.containsKey("-path")) {
      basePath = argMap.get("-path");
    }
    System.out.println("Using base path: " + basePath);

    // The path to the assignment data
    if (argMap.containsKey("-tweets")) {
      tweetPath = argMap.get("-tweets");
    }
    System.out.println("Using tweets base path: " + tweetPath);

    
    // Whether to use the validation or test set
    if (argMap.containsKey("-test")) {
      String testString = argMap.get("-test");
      if (testString.equalsIgnoreCase("test"))
        useValidation = false;
    }
    System.out.println("Testing on: " + (useValidation ? "validation" : "test"));

    // Whether or not to print the individual errors.
    if (argMap.containsKey("-verbose")) {
      verbose = true;
    }

    // Read in data
    System.out.print("Loading training sentences...");
    List<TaggedSentence> trainTaggedSentences = readTaggedSentences(basePath, 200, 2199);
    List<TaggedSentence> trainTaggedSentencesTwitter = readTaggedSentencesTwitter(tweetPath, 1, 380);

    List<TaggedSentence> devTaggedSentences = readTaggedSentences(basePath, 2200, 2299);

    
//    trainTaggedSentences.addAll(trainTaggedSentencesTwitter);
    Set<String> trainingVocabulary = extractVocabulary(trainTaggedSentences);
    
    System.out.println("done.");
    System.out.print("Loading validation sentences...");
    List<TaggedSentence> validationTaggedSentences = readTaggedSentences(basePath, 2210, 2299);
    List<TaggedSentence> validationTaggedSentencesTwitter = readTaggedSentencesTwitter(tweetPath, 381, 400);

//    validationTaggedSentences.addAll(validationTaggedSentencesTwitter);
    
    System.out.println("done.");
    System.out.print("Loading test sentences...");
    List<TaggedSentence> testTaggedSentences = readTaggedSentences(basePath, 2300, 2399);
    List<TaggedSentence> testTaggedSentencesTwitter = readTaggedSentencesTwitter(tweetPath, 401, 780);

    System.out.println("done.");

    // Construct tagger components
    // TODO : improve on the MostFrequentTagScorer
    LocalTrigramScorer localTrigramScorer = new HMMTagScorer(false);
    // TODO : improve on the GreedyDecoder
    TrellisDecoder<State> trellisDecoder = new ViterbiDecoder<State>();

    // Train tagger
    POSTagger posTagger = new POSTagger(localTrigramScorer, trellisDecoder);
    posTagger.train(trainTaggedSentences);
    posTagger.validate(devTaggedSentences);
    System.out.println("done validating");
    // Evaluation set, use either test of validation (for dev)
    final List<TaggedSentence> evalTaggedSentences;
    if (useValidation) {
    	evalTaggedSentences = validationTaggedSentences;
    } else {
    	evalTaggedSentences = testTaggedSentences;
    }
    
    // Test tagger
    evaluateTagger(posTagger, evalTaggedSentences, trainingVocabulary, verbose);
  }
}
