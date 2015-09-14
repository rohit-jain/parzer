package edu.berkeley.nlp.util;

import java.io.Serializable;
import java.util.Comparator;

import com.sun.org.apache.bcel.internal.classfile.Code;


/**
 * A generic-typed pair of objects.
 * @author Dan Klein
 */
public class Pair<F,S> implements Serializable {
  F first;
  S second;

  public static class LexicographicPairComparator<F,S>  implements Comparator<Pair<F,S>> {
    Comparator<F> firstComparator;
    Comparator<S> secondComparator;

    public int compare(Pair<F, S> pair1, Pair<F, S> pair2) {
      int firstCompare = firstComparator.compare(pair1.getFirst(), pair2.getFirst());
      if (firstCompare != 0)
        return firstCompare;
      return secondComparator.compare(pair1.getSecond(), pair2.getSecond());
    }

    public LexicographicPairComparator(Comparator<F> firstComparator, Comparator<S> secondComparator) {
      this.firstComparator = firstComparator;
      this.secondComparator = secondComparator;
    }
  }


  public F getFirst() {
    return first;
  }

  public S getSecond() {
    return second;
  }

  public void setFirst(F pFirst) {
    first = pFirst;
  }

  public void setSecond(S pSecond) {
    second = pSecond;
  }


  public boolean equals(Object o) {
    if (this == o) return true;
    if (!(o instanceof Pair)) return false;

    final Pair pair = (Pair) o;

    if (first != null ? !first.equals(pair.first) : pair.first != null) return false;
    if (second != null ? !second.equals(pair.second) : pair.second != null) return false;

    return true;
  }

  public int hashCode() {
    int result;
    result = (first != null ? first.hashCode() : 0);
    result = 29 * result + (second != null ? second.hashCode() : 0);
    return result;
  }

  public String toString() {
    return "(" + getFirst() + ", " + getSecond() + ")";
  }

  public Pair(F first, S second) {
    this.first = first;
    this.second = second;
  }
  
  /**
   * Convenience method for construction of a <code>Pair</code> with
   * the type inference on the arguments. So for instance we can type  
   *     <code>Pair<Tree<String>, Double> treeDoublePair = makePair(tree, count);</code>
   *  instead of,
   *   	 <code>Pair<Tree<String>, Double> treeDoublePair = new Pair<Tree<String>, Double>(tree, count);</code>
   * @author Aria Haghighi
   * @param <F>
   * @param <S>
   * @param f
   * @param s
   * @return <code>Pair<F,S></code> with the arguments <code>f</code>  and <code>s</code>
   */
  public static <F,S> Pair<F,S> makePair(F f, S s) {
	  return new Pair<F,S>(f,s);
  }
}

