package edu.berkeley.nlp.util;

/**
 * Convenience class that allows us to time the duration between events. The basic usage is as follows:
 *     <code>
 *      <br><br> Stopwatch parseWatch = new Stopwatch();   	   
 * 		<br>for (List&lt;String&gt; sentence: corpus) {
 * 		<br>&nbsp;  	 parseWatch.start();
 * 		<br>&nbsp;       Tree<String> parse = parser.parse(sentence);
 * 		<br>&nbsp;	     parseWatch.stop();
 * 		<br>&nbsp;       System.out.printf("Took %.3f seconds to parse.\n", parseWatch.getLastElapsedTime());
 * 		<br>}
 * 		<br> double totalParsingTimeInSecs = parseWatch.getTotalElapsedTime();
 * 		<br> System.out.printf("Parsed %d sentences in %.3f seconds\n", corpus.size(), totalParsingTimeInSecs);
 *   </code></p>
 * @author aria42
 *
 */
public class Stopwatch {

	private long startTick, stopTick;
	private double totalElapsedTime;
	private double lastElapsedTime;
	private boolean isRunning;
	
	public Stopwatch() {
		this.isRunning = false;		
		this.totalElapsedTime = 0.0;
		start();
	}
	
	public void start() {
		if (isRunning) {
			return;
		}
		this.startTick = System.currentTimeMillis();
		this.isRunning = true;
	}
	
	public void stop() {
		if (!isRunning) {
			return;
		}
		this.stopTick = System.currentTimeMillis();
		this.isRunning = false;
		double elapsedTime = (this.stopTick-this.startTick) / 1000.0;
		this.totalElapsedTime += elapsedTime;
		this.lastElapsedTime = elapsedTime;
	}
	/**
	 * Returns total time the stopwatch has been running in seconds 
	 * @return
	 */
	public double getTotalElapsedTime() {
		if (isRunning) {
			// Add time since last stop()
			stop();			
			start();
		}
		return totalElapsedTime;
	}
	/**
	 * 
	 * @return
	 */
	public double getLastElapsedTime() {
		if (isRunning) {
			stop();
		}
		return lastElapsedTime;
	}
	
	
	
	/**
	 * 
	 * @return
	 */
	public boolean isRunning() {
		return isRunning;
	}
	
	public void reset() {
		totalElapsedTime = 0.0;
		isRunning = false;		
	}
	
		
		
}
