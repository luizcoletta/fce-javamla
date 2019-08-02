import java.util.ArrayList;

import weka.clusterers.SimpleKMeans;
import weka.core.Instances;

public class ClustererOMREnsThread {

	private ArrayList<Integer> nClusters = null;
	private int nInit = 50;
	private Instances data = null;

	private double ssBestQuality = Double.NEGATIVE_INFINITY;
	private Instances clusters = null;
	private SimpleKMeans bestModel = null;
	private int bestK = -1;

	private int[] cluLabels = null;

	public ClustererOMREnsThread(ArrayList<Integer> nClusters, int nInit, Instances data){
		this.nClusters = nClusters;
		this.nInit = nInit;
		this.data = data;
		cluLabels = new int[data.numInstances()];
	}

	public void Run(){  

		try {

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
			//System.out.println("\n-> Assembling results:\n");
			for (ClustererThread t : threads){  
				//System.out.println("-> Quality - k=" + t.getK() + ": " + t.getSSBestQuality());
				if (t.getSSBestQuality() > ssBestQuality){
					ssBestQuality = t.getSSBestQuality();
					bestK = t.getK();
					clusters = t.getClusters();
					bestModel = new SimpleKMeans();
					bestModel = t.getBestModel();
				}
			}  

			if (bestModel != null){
				int maxLabel = -1;
				for (int j = 0; j < data.numInstances(); j++) {
					cluLabels[j] = bestModel.clusterInstance(data.instance(j));
					if (cluLabels[j] > maxLabel){
						maxLabel = cluLabels[j];
					}
				}
			}
			// System.out.println("\n-> Max Label Best Clustering: " + maxLabel + "\n");

		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
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

	public int[] getCluLabels() {
		return cluLabels;
	}
}
