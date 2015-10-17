package edu.berkeley.nlp.util;

import java.util.List;
import java.util.ArrayList;
import java.util.Random;

import edu.berkeley.nlp.math.SloppyMath;

/**
 * @author Dan Klein
 */
public class Counters {
  public static <E> Counter<E> normalize(Counter<E> counter) {
    Counter<E> normalizedCounter = new Counter<E>();
    double total = counter.totalCount();
    for (E key : counter.keySet()) {
      normalizedCounter.setCount(key, counter.getCount(key) / total);
    }
    return normalizedCounter;
  }

  public static <K,V> CounterMap<K,V> conditionalNormalize(CounterMap<K,V> counterMap) {
    CounterMap<K,V> normalizedCounterMap = new CounterMap<K,V>();
    for (K key : counterMap.keySet()) {
      Counter<V> normalizedSubCounter = normalize(counterMap.getCounter(key));
      for (V value : normalizedSubCounter.keySet()) {
        double count = normalizedSubCounter.getCount(value);
        normalizedCounterMap.setCount(key, value, count);
      }
    }
    return normalizedCounterMap;
  }

  public static <E> String toBiggestValuesFirstString(Counter<E> c) {
    return c.asPriorityQueue().toString();
  }

  public static <E> String toBiggestValuesFirstString(Counter<E> c, int k) {
    PriorityQueue<E> pq = c.asPriorityQueue();
    PriorityQueue<E> largestK = new FastPriorityQueue<E>();
    while (largestK.size() < k && pq.hasNext()) {
      double firstScore = pq.getPriority();
      E first = pq.next();
      largestK.setPriority(first, firstScore);
    }
    return largestK.toString();
  }

  public static <E> List<E> sortedKeys(Counter<E> counter) {
    List<E> sortedKeyList = new ArrayList<E>();
    PriorityQueue<E> pq = counter.asPriorityQueue();
    while (pq.hasNext()) {
      sortedKeyList.add(pq.next());
    }
    return sortedKeyList;
  }

  /**
   * 
   * @param <E>
   * @param x
   * @param y
   * @return
   */
  public static<E> double jensenShannonDivergence(Counter<E> x, Counter<E> y) {
	  double sum = 0.0;
	  double xTotal = x.totalCount();
	  double yTotal = y.totalCount();
	  for (E key: x.keySet()) {
		//x -> x+y/2
		double xVal = x.getCount(key) / xTotal;
		double yVal = y.getCount(key) / yTotal;
		double avg = 0.5 * (xVal + yVal);
		sum += xVal * Math.log(xVal / avg);
	  }
	  for (E key: y.keySet()) {
			//y -> x+y/2
			double xVal = x.getCount(key)/ xTotal ;
			double yVal = y.getCount(key) / yTotal;
			double avg = 0.5 * (xVal + yVal);
			sum += yVal * Math.log(yVal / avg);
	  }
	  return sum / 0.5;
  }
 
  /**
   * Simple sparse dot product method. Try to put the sparser <code>Counter</code> as the <code>x</code>
   * parameter since we iterate over those keys and search for them in the <code>y</code> parameter.
   *
   * @param x
   * @param y
   * @return dotProduct 
   */
  public static <E> double dotProduct(Counter<E> x, Counter<E> y) {
  	  double total = 0.0;
  		for (E keyX: x.keySet()){
  	  	  total += x.getCount(keyX) * y.getCount(keyX);
  	  }
  		return total;
  }
  
  private static final Random random = new Random();
  
  public static <E> E sample(Counter<E> counter) {
	  double total = counter.totalCount();
	  double rand = random.nextDouble();
	  double sum = 0.0;
	  if (total <= 0.0) {
		  throw new RuntimeException("Non-positive counter total: " + total);
	  }
	  for (E key: counter.keySet()) {
		  double count = counter.getCount(key);
		  if (count < 0.0) {
			  throw new RuntimeException("Negative count in counter: " + key + " => " + count);
		  }
		  double prob = count / total;		  
		  sum += prob;
		  if (rand < sum) {
			  return key;
		  }
	  }
	  throw new RuntimeException("Shouldn't Reach Here");
  }
}