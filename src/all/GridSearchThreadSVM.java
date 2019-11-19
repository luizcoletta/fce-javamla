package all;

import java.util.ArrayList;

import weka.core.Instances;

public class GridSearchThreadSVM {

	private Instances trainData = null;
	private Instances testData = null;
	private String properties = null;
	private double[] rangeC = {Math.pow(2,-5),Math.pow(2,-3),Math.pow(2,-1),Math.pow(2,1),Math.pow(2,3),Math.pow(2,5),Math.pow(2,7),Math.pow(2,9),Math.pow(2,11),Math.pow(2,13),Math.pow(2,15)};
	private double[] rangeG = {Math.pow(2,-15),Math.pow(2,-13),Math.pow(2,-11),Math.pow(2,-9),Math.pow(2,-7),Math.pow(2,-5),Math.pow(2,-3),Math.pow(2,-1),Math.pow(2,1),Math.pow(2,3)};
	private double bestCost = -1;
	private double bestGamma = -1;
	private double bestFit = -1;

	public GridSearchThreadSVM (Instances trainData, Instances testData, String properties) {
		this.trainData = trainData;
		this.testData = testData;
		this.properties = properties;
	}

	public void runGridSearch() throws InterruptedException {

		ArrayList<ThreadSVM> threads = new ArrayList<ThreadSVM>();  

		for (double cost : rangeC){
			for (double gamma : rangeG){
				ThreadSVM task = new ThreadSVM(trainData, testData, cost, gamma, properties);  
				threads.add(task); 
			}
		}

		// start the threads  
		for (ThreadSVM t : threads){  
			t.start();  
		}  
		for (ThreadSVM t : threads){  
			t.join();  
		}
		
		/*for (SVMThread t : threads){  
			t.run();
		}*/
		
		double fitness = -1;
		for (ThreadSVM t : threads){  
			if (t.getAccuracy()>fitness){
				fitness = t.getAccuracy();
				bestCost = t.getCost();
				bestGamma = t.getGamma();
				bestFit = fitness;
			}
		}  
	}

	public double getBestCost() {
		return bestCost;
	}

	public double getBestGamma() {
		return bestGamma;
	}

	public double getBestFit() {
		return bestFit;
	}
}
