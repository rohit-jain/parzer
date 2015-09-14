package edu.berkeley.nlp.util;

/**
 * Convenience Extension of Counter to use an IdentityHashMap.
 *
 * @author Dan Klein
 */
public class IdentityCounter<E> extends Counter<E> {
  public IdentityCounter() {
    super(new MapFactory.IdentityHashMapFactory<E,Double>());
  }
}
