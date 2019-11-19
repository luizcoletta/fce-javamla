package all;

import java.util.ArrayList;

import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class ClustererOMRThread {
    
	private ArrayList<Integer> nClusters = null;
	private int nInit = 50;
	private Instances data = null;
	
	private double ssBestQuality = Double.NEGATIVE_INFINITY;
	private Instances clusters = null;
	private SimpleKMeans bestModel = null;
	private int bestK = -1;
        
        private int[] cluLabels = null;
	
	public ClustererOMRThread(ArrayList<Integer> nClusters, int nInit, Instances data){
		this.nClusters = nClusters;
		this.nInit = nInit;
		this.data = data;
	}

	public int Run() throws InterruptedException{  
		  
		ArrayList<ClustererThread> threads = new ArrayList<ClustererThread>();  

		for (int c : nClusters){
			ClustererThread task = new ClustererThread(c, nInit, data);  
			threads.add(task); 
		}

		// start threads  
		for (ClustererThread t : threads){  
			t.start();  
		}  
		for (ClustererThread t : threads){  
			t.join();  
		}  
		
		// joining results
		System.out.println("\n-> Assembling results:\n");
		for (ClustererThread t : threads){  
			System.out.println("-> Quality - k=" + t.getK() + ": " + t.getSSBestQuality());
			if (t.getSSBestQuality() > ssBestQuality){
				ssBestQuality = t.getSSBestQuality();
				bestK = t.getK();
				clusters = t.getClusters();
				bestModel = new SimpleKMeans();
				bestModel = t.getBestModel();
                                
			}
		}  
		
		return bestK;
	}

	public double getSSBestQuality() {
		return ssBestQuality;
	}
	
	public int getBestK() {
		return bestK;
	}

	public SimpleKMeans getBestModel() {
	    return bestModel;
    }

	public Instances getClusters() {
		return clusters;
	}
}
