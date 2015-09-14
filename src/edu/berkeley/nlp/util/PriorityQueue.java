package edu.berkeley.nlp.util;

import java.util.Iterator;

/**
 * Priority queue interface: higher priorities are at the head of the queue.  GeneralPriorityQueue implements all of the
 * methods, while FastPriorityQueue does not support removal or promotion in the normal manner.
 *
 * @author Dan Klein
 */
public interface PriorityQueue <E> extends Iterator<E> {
  E getFirst();

  E removeFirst();

  double getPriority();

  boolean containsKey(E element);

  double removeKey(E element);

  void setPriority(E element, double priority);

  double getPriority(E element);

  String toString(int maxKeysToPrint);

  int size();

  boolean isEmpty();
}
