package edu.berkeley.nlp.util;

import java.io.Serializable;
import java.util.NoSuchElementException;

/**
 * A priority queue based on a binary heap.  Note that this implementation does not efficiently support containsKey or
 * getPriority.  Removal is not supported.  If you set the priority of a key multiple times, it will NOT be promoted or
 * demoted, but rather it will be inserted in the queue once multiple times, with the various priorities.
 *
 * @author Dan Klein
 */
public class FastPriorityQueue <E> implements PriorityQueue<E>, Serializable {
  private static final long serialVersionUID = 5724671156522771658L;
  int size;
  int capacity;
  Object[] elements;
  double[] priorities;

  public boolean containsKey(E e) {
    for (int i = 0; i < elements.length; i++) {
      Object element = elements[i];
      if (e == null && element == null) return true;
      if (e != null && e.equals(element)) return true;
    }
    return false;
  }

  public double getPriority(E e) {
    double bestPriority = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < elements.length; i++) {
      Object element = elements[i];
      if ((e == null && element == null) ||
          (e != null && e.equals(element))) {
        if (priorities[i] > bestPriority) {
          bestPriority = priorities[i];
        }
      }
    }
    return bestPriority;
  }

  public double removeKey(E e) {
    throw new UnsupportedOperationException();
  }

  protected void grow(int newCapacity) {
    Object[] newElements = new Object[newCapacity];
    double[] newPriorities = new double[newCapacity];
    if (size > 0) {
      System.arraycopy(elements, 0, newElements, 0, elements.length);
      System.arraycopy(priorities, 0, newPriorities, 0, priorities.length);
    }
    elements = newElements;
    priorities = newPriorities;
    capacity = newCapacity;
  }

  protected int parent(int loc) {
    return (loc - 1) / 2;
  }

  protected int leftChild(int loc) {
    return 2 * loc + 1;
  }

  protected int rightChild(int loc) {
    return 2 * loc + 2;
  }

  protected void heapifyUp(int loc) {
    if (loc == 0) return;
    int parent = parent(loc);
    if (priorities[loc] > priorities[parent]) {
      swap(loc, parent);
      heapifyUp(parent);
    }
  }

  protected void heapifyDown(int loc) {
    int max = loc;
    int leftChild = leftChild(loc);
    if (leftChild < size()) {
      double priority = priorities[loc];
      double leftChildPriority = priorities[leftChild];
      if (leftChildPriority > priority)
        max = leftChild;
      int rightChild = rightChild(loc);
      if (rightChild < size()) {
        double rightChildPriority = priorities[rightChild(loc)];
        if (rightChildPriority > priority && rightChildPriority > leftChildPriority)
          max = rightChild;
      }
    }
    if (max == loc)
      return;
    swap(loc, max);
    heapifyDown(max);
  }

  protected void swap(int loc1, int loc2) {
    double tempPriority = priorities[loc1];
    Object tempElement = elements[loc1];
    priorities[loc1] = priorities[loc2];
    elements[loc1] = elements[loc2];
    priorities[loc2] = tempPriority;
    elements[loc2] = tempElement;
  }

  public E removeFirst() {
    if (size < 1) throw new NoSuchElementException();
    Object element = elements[0];
    swap(0, size - 1);
    size--;
    elements[size] = null;
    heapifyDown(0);
    return (E) element;
  }

  /**
   * Returns true if the priority queue is non-empty
   */
  public boolean hasNext() {
    return !isEmpty();
  }

  /**
   * Returns the element in the queue with highest priority, and pops it from the queue.
   */
  public E next() {
    return removeFirst();
  }

  /**
   * Not supported -- next() already removes the head of the queue.
   */
  public void remove() {
    throw new UnsupportedOperationException();
  }

  /**
   * Returns the highest-priority element in the queue, but does not pop it.
   */
  public E getFirst() {
    if (size < 1) throw new NoSuchElementException();
    return (E) elements[0];
  }

  /**
   * Gets the priority of the highest-priority element of the queue.
   */
  public double getPriority() {
    if (size() > 0)
      return priorities[0];
    throw new NoSuchElementException();
  }

  /**
   * Number of elements in the queue.
   */
  public int size() {
    return size;
  }

  /**
   * True if the queue is empty (size == 0).
   */
  public boolean isEmpty() {
    return size == 0;
  }

  /**
   * Adds a key to the queue with the given priority.  If the key is already in the queue, it will be added an
   * additional time, NOT promoted/demoted.
   *
   * @param key
   * @param priority
   */
  public void setPriority(E key, double priority) {
    if (size == capacity) {
      grow(2 * capacity + 1);
    }
    elements[size] = key;
    priorities[size] = priority;
    heapifyUp(size);
    size++;
  }

  /**
   * Returns a representation of the queue in decreasing priority order.
   */
  public String toString() {
    return toString(size());
  }

  /**
   * Returns a representation of the queue in decreasing priority order, displaying at most maxKeysToPrint elements.
   *
   * @param maxKeysToPrint
   */
  public String toString(int maxKeysToPrint) {
    PriorityQueue<E> pq = deepCopy();
    StringBuilder sb = new StringBuilder("[");
    int numKeysPrinted = 0;
    while (numKeysPrinted < maxKeysToPrint && ! pq.isEmpty()) {
      double priority = pq.getPriority();
      E element = pq.removeFirst();
      sb.append(element.toString());
      sb.append(" : ");
      sb.append(priority);
      if (numKeysPrinted < size() - 1)
//        sb.append("\n");
        sb.append(", ");
      numKeysPrinted++;
    }
    if (numKeysPrinted < size())
      sb.append("...");
    sb.append("]");
    return sb.toString();
  }

  /**
   * Returns a counter whose keys are the elements in this priority queue, and whose counts are the priorities in this
   * queue.  In the event there are multiple instances of the same element in the queue, the counter's count will be the
   * sum of the instances' priorities.
   *
   * @return
   */
  public Counter asCounter() {
    PriorityQueue<E> pq = deepCopy();
    Counter<E> counter = new Counter<E>();
    while (! pq.isEmpty()) {
      double priority = pq.getPriority();
      E element = pq.removeFirst();
      counter.incrementCount(element, priority);
    }
    return counter;
  }

  /**
   * Returns a clone of this priority queue.  Modifications to one will not affect modifications to the other.
   */
  public FastPriorityQueue<E> deepCopy() {
    FastPriorityQueue<E> clonePQ = new FastPriorityQueue<E>();
    clonePQ.size = size;
    clonePQ.capacity = capacity;
    clonePQ.elements = new Object[capacity];
    clonePQ.priorities = new double[capacity];
    if (size() > 0) {
      System.arraycopy(elements, 0, clonePQ.elements, 0, size());
      System.arraycopy(priorities, 0, clonePQ.priorities, 0, size());
    }
    return clonePQ;
  }

  public FastPriorityQueue() {
    this(15);
  }

  public FastPriorityQueue(int capacity) {
    int legalCapacity = 0;
    while (legalCapacity < capacity) {
      legalCapacity = 2 * legalCapacity + 1;
    }
    grow(legalCapacity);
  }

  public static void main(String[] args) {
    PriorityQueue<String> pq = new FastPriorityQueue<String>();
    System.out.println(pq);
    pq.setPriority("one", 1);
    System.out.println(pq);
    pq.setPriority("three", 3);
    System.out.println(pq);
    pq.setPriority("one", 1.1);
    System.out.println(pq);
    pq.setPriority("two", 2);
    System.out.println(pq);
    System.out.println(pq.toString(2));
    while (pq.hasNext()) {
      System.out.println(pq.next());
    }
  }
}
