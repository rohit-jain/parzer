package edu.berkeley.nlp.util;

import java.io.Serializable;
import java.util.Collection;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;
import java.util.HashMap;

/**
 * A map from objects to doubles.  Includes convenience methods for getting,
 * setting, and incrementing element counts.  Objects not in the counter will
 * return a count of zero.  The counter is backed by a HashMap (unless specified
 * otherwise with the MapFactory constructor).
 *
 * @author Dan Klein
 */
public class Counter <E> implements Serializable {
  private static final long serialVersionUID = 5724671156522771655L;

  Map<E, Double> entries;

  int currentModCount = 0;
  int cacheModCount = -1;
  double cacheTotalCount = 0.0;

  /**
   * The elements in the counter.
   *
   * @return set of keys
   */
  public Set<E> keySet() {
    return entries.keySet();
  }

  /**
   * The number of entries in the counter (not the total count -- use
   * totalCount() instead).
   */
  public int size() {
    return entries.size();
  }

  /**
   * True if there are no entries in the counter (false does not mean totalCount
   * > 0)
   */
  public boolean isEmpty() {
    return size() == 0;
  }

  /**
   * Returns whether the counter contains the given key.  Note that this is the
   * way to distinguish keys which are in the counter with count zero, and those
   * which are not in the counter (and will therefore return count zero from
   * getCount().
   *
   * @param key
   * @return whether the counter contains the key
   */
  public boolean containsKey(E key) {
    return entries.containsKey(key);
  }

  /**
   * Remove a key from the counter. Returns the count associated with that
   * key or zero if the key wasn't in the counter to begin with
   * @param key
   * @return the count associated with the key
   */
  public double removeKey(E key) {
      Double d  = entries.remove(key);
      return (d == null? 0.0: d);
  }
  /**
   * Get the count of the element, or zero if the element is not in the
   * counter.
   *
   * @param key
   * @return
   */
  public double getCount(E key) {
    Double value = entries.get(key);
    if (value == null)
      return 0;
    return value;
  }

  /**
   * Get the log count of the element, or Double.NEGATIVE_INFINITY if the element is not in the
   * counter.
   *
   * @param key
   * @return
   */
  public double getMinCount(E key) {
    Double value = entries.get(key);
    if (value == null)
      return Double.NEGATIVE_INFINITY;
    return value;
  }

  
  /**
   * Set the count for the given key, clobbering any previous count.
   *
   * @param key
   * @param count
   */
  public void setCount(E key, double count) {
    currentModCount++;
    entries.put(key, count);
  }

  /**
   * Increment a key's count by the given amount.
   *
   * @param key
   * @param increment
   */
  public void incrementCount(E key, double increment) {
    setCount(key, getCount(key) + increment);
  }

  /**
   * Increment each element in a given collection by a given amount.
   */
  public void incrementAll(Collection<? extends E> collection, double count) {
    for (E key : collection) {
      incrementCount(key, count);
    }
  }

  public <T extends E> void incrementAll(Counter<T> counter) {
    for (T key : counter.keySet()) {
      double count = counter.getCount(key);
      incrementCount(key, count);
    }
  }

  public <T extends E> void elementwiseMax(Counter<T> counter) {
	  for (T key : counter.keySet()) {
		  double count = counter.getCount(key);
		  if ( getCount(key) < count ) {
			  setCount(key, count);
		  }
	  }
  }
  
  /**
   * Finds the total of all counts in the counter.  This implementation uses
   * cached count which may get out of sync if the entries map is modified in
   * some unantipicated way.
   *
   * @return the counter's total
   */
  public double totalCount() {
    if (currentModCount != cacheModCount) {
      double total = 0.0;
      for (Map.Entry<E, Double> entry : entries.entrySet()) {
        total += entry.getValue();
      }
      cacheTotalCount = total;
      cacheModCount = currentModCount;
    }
    return cacheTotalCount;
  }

  /**
   * Destructively normalize this Counter in place.
   */
  public void normalize() {
    double totalCount = totalCount();
    for (E key : keySet()) {
      setCount(key, getCount(key) / totalCount);
    }
  }

  /**
   * Destructively scale this Counter in place.
   */
  public void scale(double scaleFactor) {
    for (E key : keySet()) {
      setCount(key, getCount(key) * scaleFactor);
    }
  }

  /**
   * Return Sum of the distribution
   */
  public Double sum(){
	  Double total = 0.0;
	  for (E key: keySet()){
		  total += (getCount(key));
	  }
	  return total;
  }
  
  /**
   * Return Standard deviation of a probability distribution
   */
  public Double standardDeviation(){
	  Double mean = sum()/new Double(keySet().size());
	  Double std = 0.0;
	  for (E key: keySet()){
		  std += Math.pow((getCount(key) - mean),2);
	  }
	  System.out.printf("sum %f, mean:%f ", sum() ,mean);
	  return std/(new Double(keySet().size()) );
  }

  /**
   * Finds the key with maximum count.  This is a linear operation, and ties are
   * broken arbitrarily.
   *
   * @return a key with minumum count
   */
  public E argMax() {
    double maxCount = Double.NEGATIVE_INFINITY;
    E maxKey = null;
    for (Map.Entry<E, Double> entry : entries.entrySet()) {
      if (entry.getValue() > maxCount || maxKey == null) {
        maxKey = entry.getKey();
        maxCount = entry.getValue();
      }
    }
    return maxKey;
  }

  /**
   * Returns a string representation with the keys ordered by decreasing
   * counts.
   *
   * @return string representation
   */
  public String toString() {
    return toString(keySet().size());
  }

  /**
   * Returns a string representation which includes no more than the
   * maxKeysToPrint elements with largest counts.
   *
   * @param maxKeysToPrint
   * @return partial string representation
   */
  public String toString(int maxKeysToPrint) {
    return asPriorityQueue().toString(maxKeysToPrint);
  }

  /**
   * Builds a priority queue whose elements are the counter's elements, and
   * whose priorities are those elements' counts in the counter.
   */
  public PriorityQueue<E> asPriorityQueue() {
    PriorityQueue<E> pq = new FastPriorityQueue<E>(entries.size());
    for (Map.Entry<E, Double> entry : entries.entrySet()) {
      pq.setPriority(entry.getKey(), entry.getValue());
    }
    return pq;
  }
  
  public Map<E, Double> getTopK(int K){
	  PriorityQueue<E> pq = asPriorityQueue();
	  Map<E, Double> topElements = new HashMap<E, Double>();
	  if(K > entries.size()){
		  K = entries.size();
	  }
	  for(int i=0; i<K; ++i){
		  double count = entries.get(pq.getFirst());
		  E element = pq.removeFirst();
		  topElements.put(element, count);
	  }
	  return topElements;
  }
  
  /**
   * Entry sets are an efficient way to iterate over
   * the key-value pairs in a map
   * @return entrySet
   */
  public Set<Entry<E, Double>> getEntrySet() {
	return entries.entrySet();
  }

  public Counter() {
    this(new MapFactory.HashMapFactory<E, Double>());
  }

  public Counter(MapFactory<E, Double> mf) {
    entries = mf.buildMap();
  }

  public Counter(Counter<? extends E> counter) {
    this();
    incrementAll(counter);
  }

  public Counter(Collection<? extends E> collection) {
    this();
    incrementAll(collection, 1.0);
  }

  public static void main(String[] args) {
    Counter<String> counter = new Counter<String>();
    System.out.println(counter);
    counter.incrementCount("planets", 7);
    System.out.println(counter);
    counter.incrementCount("planets", 1);
    System.out.println(counter);
    counter.setCount("suns", 1);
    System.out.println(counter);
    counter.setCount("aliens", 0);
    System.out.println(counter);
    System.out.println(counter.toString(2));
    System.out.println("Total: " + counter.totalCount());
  }

}
