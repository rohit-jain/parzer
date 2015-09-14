package edu.berkeley.nlp.util;

import java.io.*;
import java.util.*;

/**
 * Maintains a two-way map between a set of objects and contiguous integers from
 * 0 to the number of objects.  Use get(i) to look up object i, and
 * indexOf(object) to look up the index of an object.
 *
 * @author Dan Klein
 */
public class Indexer <E> extends AbstractList<E> implements Serializable {
  private static final long serialVersionUID = -8769544079136550516L;
  List<E> objects;
  Map<E, Integer> indexes;

  /**
   * Return the object with the given index
   *
   * @param index
   */
  public E get(int index) {
    return objects.get(index);
  }

  /**
   * Returns the number of objects indexed.
   */
  public int size() {
    return objects.size();
  }

  /**
   * Returns the index of the given object, or -1 if the object is not present
   * in the indexer.
   *
   * @param o
   * @return
   */
  public int indexOf(Object o) {
    Integer index = indexes.get(o);
    if (index == null)
      return -1;
    return index;
  }

  /**
   * Add an element to the indexer if not already present. In either case,
   * returns the index of the given object.
   * @param e
   * @return
   */
  public int addAndGetIndex(E e) {
    Integer index = indexes.get(e);
    if (index != null) {
   	 return index;
    }
    //  Else, add
    int newIndex = size();
    objects.add(e);
    indexes.put(e,newIndex); 
    return newIndex;
  }
  
  /**
   * Constant time override for contains.
   */
  public boolean contains(Object o) {
    return indexes.keySet().contains(o);
  }

  /**
   * Add an element to the indexer.  If the element is already in the indexer,
   * the indexer is unchanged (and false is returned).
   *
   * @param e
   * @return
   */
  public boolean add(E e) {
    if (contains(e)) return false;
    objects.add(e);
    indexes.put(e, size() - 1);
    return true;
  }

  public Indexer() {
    objects = new ArrayList<E>();
    indexes = new HashMap<E, Integer>();
  }
  
  public Indexer(Collection<? extends E> c) {
    this();
    addAll(c);
  }
}
