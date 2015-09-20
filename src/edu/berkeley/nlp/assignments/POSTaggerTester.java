package edu.berkeley.nlp.assignments;

import edu.berkeley.nlp.io.PennTreebankReader;
import edu.berkeley.nlp.ling.Tree;
import edu.berkeley.nlp.ling.Trees;
import edu.berkeley.nlp.util.*;

import java.util.*;

/**
 * @author Dan Klein
 */
public class POSTaggerTester {

  static final String START_WORD = "<S>";
  static final String STOP_WORD = "</S>";
  static final String START_TAG = "<S>";
  static final String STOP_TAG = "</S>";
  static final String UNKNOWN = "<UNK>";
  static final int SUFFIX_LEN = 4;

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
        tags.add(states.get(0).getPreviousPreviousTag());
        for (State state : states) {
          tags.add(state.getPreviousTag());
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
	      Set<S> previousStates = new HashSet<S>();
	      Set<S> nextStates = new HashSet<S>();
	      Set<S> allStates = new HashSet<S>();
	      List<Map<S,S>> backPointers = new ArrayList<Map<S,S>>();
	      
	      CounterMap< Integer , S> pie = new CounterMap< Integer , S>();
	      
	      S currentState = trellis.getStartState();
	      
	      // set position to 0
	      int position = 0;
	      allStates.add(currentState);
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
	      
//	      System.out.println(allStates);
	      position = 0;
	      // set the start state probability to 1 for position 0
	      pie.setCount(position, currentState, 0.0);
	      
	      position += 1;
	      
	      while( position < sentenceLength+3 ){
//	        System.out.println(position);
	      	Set<S> tempStates = new HashSet<S>();
	        Map<S,S> stateBackPointers = new HashMap<S,S>();
	        for(S s: allStates){
	        	Counter<S> viterbiTransitions = new Counter<S>();
	        	Counter<S> transitions = trellis.getForwardTransitions(s);
	        	//tempStates.addAll(transitions.keySet());
//        		System.out.println("outer loop");
//	        	System.out.println(s.toString());
//        		System.out.println("inner loop next");
	  	      
	        	for(S s_i: allStates){
	        		if (pie.getCounter(position - 1).keySet().contains(s_i)){
	        			double oldPie = pie.getMinCount(position-1 , s_i);
	        			double pq;
	        			if (trellis.getForwardTransitions(s_i).keySet().contains(s)){
		        			pq = trellis.getForwardTransitions(s_i).getCount(s);
			        		double p = oldPie + pq ;
				        	viterbiTransitions.setCount(s_i, p);
		        		}
	        		}
	        	}
	        	S maxState = viterbiTransitions.argMax();
	        	stateBackPointers.put(s, maxState);
	        	pie.setCount(position, s, viterbiTransitions.getCount(maxState));
//	        	System.out.println("bestState");
//	        	System.out.println(viterbiTransitions.getCount(maxState));
//	        	System.out.println(maxState);
	        }
	        backPointers.add(position, stateBackPointers);
//	        previousStates = nextStates;
//	        nextStates = tempStates;
	        position += 1;
	      }
	      
//	      System.out.println("Done calculating pie");
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
//	      System.out.println("Returning States");
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

    public String toString() {
      return "[" + getPreviousPreviousTag() + ", " + getPreviousTag() + ", " + getCurrentWord() + "]";
    }

    public LocalTrigramContext(List<String> words, int position, String previousPreviousTag, String previousTag) {
      this.words = words;
      this.position = position;
      this.previousTag = previousTag;
      this.previousPreviousTag = previousPreviousTag;
    }
  }

  /**
   * A LabeledLocalTrigramContext is a context plus the correct tag for that
   * position -- basically a LabeledFeatureVector
   */
  static class LabeledLocalTrigramContext extends LocalTrigramContext {
    String currentTag;

    public String getCurrentTag() {
      return currentTag;
    }

    public String toString() {
      return "[" + getPreviousPreviousTag() + ", " + getPreviousTag() + ", " + getCurrentWord() + "_" + getCurrentTag() + "]";
    }

    public LabeledLocalTrigramContext(List<String> words, int position, String previousPreviousTag, String previousTag, String currentTag) {
      super(words, position, previousPreviousTag, previousTag);
      this.currentTag = currentTag;
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

    CounterMap<String, String> tagsToTags = new CounterMap<String, String>();
    CounterMap<String, String> previousTagToTags = new CounterMap<String, String>();
    CounterMap<String, String> tagsToWords = new CounterMap<String, String>();
    CounterMap<String, String> suffixToTags = new CounterMap<String, String>();
    CounterMap<String, String> tagsToSuffix = new CounterMap<String, String>();
    CounterMap<String, String> smoothedSuffixToTags = new CounterMap<String, String>();
    CounterMap<String, String> wordToTags = new CounterMap<String, String>();
    
    Counter<String> unknownPrecedingTags = new Counter<String>();
    Counter<String> unknownTagWords = new Counter<String>();

    Counter<String> knownSuffixes = new Counter<String>();
    Counter<String> knownWords = new Counter<String>();
    Set<String> seenTagTrigrams = new HashSet<String>();
    
    Counter<String> infrequentTags = new Counter<String>();
    Counter<String> t3 = new Counter<String>();
    Counter<String> t2 = new Counter<String>();
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

    public Double smoothTransitionProbability(CounterMap<String, String> trigramCounter, String tag, String previousTag, String precedingTags, String bi_tag){
    	Double p = 0.0;
    	Double p_t3 = t3.getCount(tag)/t3.totalCount();
    	Double p_t3t2 = t2t3.getCount(bi_tag)/t2.getCount(previousTag);
    	if (p_t3t2 != previousTagToTags.getCount(previousTag, tag))
    		System.out.println("mismatch");
    	Double p_t1t2t3 = trigramCounter.getCount(precedingTags, tag);
    	p = (lambda1 * p_t3) + (lambda2 * p_t3t2) + (lambda3 * p_t1t2t3);
    	return p;
    }
    
    public Counter<String> getLogScoreCounter(LocalTrigramContext localTrigramContext) {
        int position = localTrigramContext.getPosition();
        String word = localTrigramContext.getWords().get(position);
        String precedingTags = makeBigramString(localTrigramContext.getPreviousPreviousTag(), localTrigramContext.getPreviousTag());
        String previousTag = localTrigramContext.getPreviousTag();
        String previousPreviousTag = localTrigramContext.getPreviousPreviousTag();

        Counter<String> precedingTagCounter;
        Counter<String> logScoreCounter = new Counter<String>();
        
        
        if (word == STOP_WORD){
    	  double transition_probability = smoothTransitionProbability(tagsToTags, STOP_TAG, previousTag, precedingTags, makeBigramString(previousTag, STOP_TAG));
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
        	Set<String> candidateTags = wordToTags.getCounter(word).keySet();
        	for (String candidateTag : candidateTags) {
        	  double transition_probability = smoothTransitionProbability(tagsToTags, candidateTag, previousTag, precedingTags, makeBigramString(previousTag, candidateTag));
        	  double emission_probability = tagsToWords.getCount(candidateTag,word);
        	  
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
//        	System.out.println("unknown");

        	String bestSuffix = getSuffix(word, SUFFIX_LEN);
        	for(int i=SUFFIX_LEN; i > 0; i--){
        		String newSuffix = getSuffix(word, i);
        		if (knownSuffixes.containsKey(newSuffix)){
        			bestSuffix = newSuffix;
        			break;
        		}
        	}
    		// get candidate tags
    		Set<String> candidateTags = smoothedSuffixToTags.getCounter(bestSuffix).keySet();
    		// get probability for each candidate tag
    		for (String candidateTag : candidateTags) {
          	  double transition_probability = smoothTransitionProbability(tagsToTags, candidateTag, previousTag, precedingTags, makeBigramString(previousTag, candidateTag));
          	  double emission_probability = tagsToSuffix.getCount(candidateTag, bestSuffix);// * tagsToWords.getCount(candidateTag, UNKNOWN);

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

    
    private Set<String> allowedFollowingTags(Set<String> tags, String previousPreviousTag, String previousTag) {
        Set<String> allowedTags = new HashSet<String>();
        for (String tag : tags) {
          String trigramString = makeTrigramString(previousPreviousTag, previousTag, tag);
          if (seenTagTrigrams.contains((trigramString))) {
            allowedTags.add(tag);
          }
        }
        return allowedTags;
    }
    
    private String makeBigramString(String a, String b){
        return a + " " + b;
    }
    
    private String makeTrigramString(String previousPreviousTag, String previousTag, String currentTag) {
      return previousPreviousTag + " " + previousTag + " " + currentTag;
    }
    
    public Double pHat(int l, String s, String t, double theta){
    	if(l == 0){
    		// t3 is basically known tags
    		return infrequentTags.getCount(t);
    	}
    	return (suffixToTags.getCounter(getSuffix(s, l)).getCount(t) + (theta * pHat(l-1, s, t, theta)) )/( 1 + theta);
    }

    public void train(List<LabeledLocalTrigramContext> labeledLocalTrigramContexts) {
      Double theta = 0.0;
      int maxLen = 0;
      // collect word-tag counts
      for (LabeledLocalTrigramContext labeledLocalTrigramContext : labeledLocalTrigramContexts) {
        String word = labeledLocalTrigramContext.getCurrentWord();
        String tag = labeledLocalTrigramContext.getCurrentTag();
        String previousTag = labeledLocalTrigramContext.getPreviousTag();
        String previousPreviousTag = labeledLocalTrigramContext.getPreviousPreviousTag();
        String precedingTags = makeBigramString(previousTag, previousPreviousTag);
        String trigram = makeTrigramString(labeledLocalTrigramContext.getPreviousPreviousTag(), labeledLocalTrigramContext.getPreviousTag(), labeledLocalTrigramContext.getCurrentTag());
        if(word.length()>maxLen){
        	maxLen = word.length();
        }
        
        if (!tagsToTags.keySet().contains(precedingTags))
        {
        	unknownPrecedingTags.incrementCount(tag, 1.0);
        }
        if (!tagsToWords.keySet().contains(tag)){
        	unknownTagWords.incrementCount(word, 1.0);
        }
        
		tagsToTags.incrementCount(precedingTags, tag, 1.0);
        tagsToWords.incrementCount(tag, word, 1.0);
        wordToTags.incrementCount(word, tag, 1.0);
        previousTagToTags.incrementCount(previousTag, tag, 1.0);
        seenTagTrigrams.add(trigram);
        
        knownWords.incrementCount(word, 1.0);
        
        t2.incrementCount(previousTag, 1.0);
        t3.incrementCount(tag,1.0);
        t1t2.incrementCount(precedingTags, 1.0);
        t2t3.incrementCount(makeBigramString(previousTag, tag), 1.0);
        t1t2t3.incrementCount(trigram, 1.0);
        
      }
      
      System.out.println(maxLen);
      
      for (LabeledLocalTrigramContext labeledLocalTrigramContext : labeledLocalTrigramContexts) {
          String word = labeledLocalTrigramContext.getCurrentWord();
          String tag = labeledLocalTrigramContext.getCurrentTag();
          if(knownWords.getCount(word) < 6){
	          for( int i=1; i <= Math.min(SUFFIX_LEN, word.length()); i++){
	  			String suffix = getSuffix(word, i);
	  			suffixToTags.incrementCount(suffix, tag, 1.0);
	  			knownSuffixes.incrementCount(suffix, 1.0);
	          }
          }
          infrequentTags.incrementCount(tag, 1.0);
      }
      
      // TODO: add one unknown to every tag
      for (String tag: tagsToWords.keySet()){
    	  tagsToWords.incrementCount(tag, UNKNOWN, 1.0);
      }

      tagsToTags = Counters.conditionalNormalize(tagsToTags);
      tagsToWords = Counters.conditionalNormalize(tagsToWords);
      previousTagToTags = Counters.conditionalNormalize(previousTagToTags);
      suffixToTags = Counters.conditionalNormalize(suffixToTags);
      
      unknownTagWords = Counters.normalize(unknownTagWords);
      unknownPrecedingTags = Counters.normalize(unknownPrecedingTags);
      knownSuffixes = Counters.normalize(knownSuffixes);
      infrequentTags = Counters.normalize(infrequentTags);
      
      theta = infrequentTags.standardDeviation();
      System.out.println(theta);
      
      // smoothed suffix probabilities
      for (String s: suffixToTags.keySet()){
    	for (String t: suffixToTags.getCounter(s).keySet()){
    		// different distribution to protect old calc
    		smoothedSuffixToTags.setCount(s, t, pHat(s.length(), s, t, theta));
    	}  
      }
      
      // apply bayes rule to get tags to suffix
      for(String s: smoothedSuffixToTags.keySet()){
    	  for(String t: smoothedSuffixToTags.getCounter(s).keySet()){
    		  Double p = (smoothedSuffixToTags.getCounter(s).getCount(t) * knownSuffixes.getCount(s))/infrequentTags.getCount(t);
    		  tagsToSuffix.incrementCount(t, s, p);
    	  }
      }
    }

    public void validate(List<LabeledLocalTrigramContext> labeledLocalTrigramContexts) {
      // tune using linear interpolation
    	Set<String> seenSmoothingTrigrams = new HashSet<String>();
      for (LabeledLocalTrigramContext labeledLocalTrigramContext : labeledLocalTrigramContexts) {
          String word = labeledLocalTrigramContext.getCurrentWord();
          String tag = labeledLocalTrigramContext.getCurrentTag();
          String previousTag = labeledLocalTrigramContext.getPreviousTag();
          String previousPreviousTag = labeledLocalTrigramContext.getPreviousPreviousTag();
          String precedingTags = makeBigramString(previousTag, previousPreviousTag);
          String trigram = makeTrigramString(labeledLocalTrigramContext.getPreviousPreviousTag(), labeledLocalTrigramContext.getPreviousTag(), labeledLocalTrigramContext.getCurrentTag());
    	  
          Double fif2f3 = t1t2t3.getCount(trigram);
          Double f1f2 = t1t2.getCount(precedingTags);
          Double f2f3 = t2t3.getCount(makeBigramString(previousTag, tag));
          Double f2 = t2.getCount(previousTag);
          Double f3 = t3.getCount(tag);
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

  private static void evaluateTagger(POSTagger posTagger, List<TaggedSentence> taggedSentences, Set<String> trainingVocabulary, boolean verbose) {
    double numTags = 0.0;
    double numTagsCorrect = 0.0;
    double numUnknownWords = 0.0;
    double numUnknownWordsCorrect = 0.0;
    int numDecodingInversions = 0;
    int count = 0;
    for (TaggedSentence taggedSentence : taggedSentences) {
      List<String> words = taggedSentence.getWords();
      List<String> goldTags = taggedSentence.getTags();
      List<String> guessedTags = posTagger.tag(words);
      for (int position = 0; position < words.size() - 1; position++) {
        String word = words.get(position);
        String goldTag = goldTags.get(position);
        String guessedTag = guessedTags.get(position);
        if (guessedTag.equals(goldTag))
          numTagsCorrect += 1.0;
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
        System.out.println("suboptimal");
        System.out.println(scoreOfGoldTagging);
        System.out.println(scoreOfGuessedTagging);
        
        if (verbose) System.out.println("WARNING: Decoder suboptimality detected.  Gold tagging has higher score than guessed tagging.");
      }
//      else{
//          System.out.println("not suboptimal");
//          System.out.println(scoreOfGoldTagging);
//          System.out.println(scoreOfGuessedTagging);
//    	  
//      }
      if (verbose) System.out.println(alignedTaggings(words, goldTags, guessedTags, true) + "\n");
      //break after one sentence
      //break;
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
    boolean verbose = false;
    boolean useValidation = true;

    // Update defaults using command line specifications

    // The path to the assignment data
    if (argMap.containsKey("-path")) {
      basePath = argMap.get("-path");
    }
    System.out.println("Using base path: " + basePath);

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
    Set<String> trainingVocabulary = extractVocabulary(trainTaggedSentences);
    System.out.println("done.");
    System.out.print("Loading validation sentences...");
    List<TaggedSentence> validationTaggedSentences = readTaggedSentences(basePath, 2200, 2299);
    System.out.println("done.");
    System.out.print("Loading test sentences...");
    List<TaggedSentence> testTaggedSentences = readTaggedSentences(basePath, 2300, 2399);
    System.out.println("done.");

    // Construct tagger components
    // TODO : improve on the MostFrequentTagScorer
    LocalTrigramScorer localTrigramScorer = new HMMTagScorer(false);
    // TODO : improve on the GreedyDecoder
    TrellisDecoder<State> trellisDecoder = new ViterbiDecoder<State>();

    // Train tagger
    POSTagger posTagger = new POSTagger(localTrigramScorer, trellisDecoder);
    posTagger.train(trainTaggedSentences);
    posTagger.validate(validationTaggedSentences);
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
