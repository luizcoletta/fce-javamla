
import java.util.Random;
import weka.clusterers.SimpleKMeans;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instances;

public class ClustererThread extends Thread{ 

	private int k = 2;
	private int nInit = 50;
	private Instances data = null;

	private double ssBestQuality = Double.NEGATIVE_INFINITY;
	private Instances clusters = null;
	private SimpleKMeans bestModel = null;

	private int[] cluLabels = null;

	public ClustererThread(int k, int nInit, Instances data){
		this.k = k;
		this.nInit = nInit;
		this.data = data;
		cluLabels = new int[data.numInstances()];
	}

	public void run() {

		//System.out.println("-> Running clusterer thread " + Thread.currentThread().getId() + " (k=" + k + ")");

		double ssQuality = 0;

		try {
			for (int i=0; i<nInit; ++i){
                               
                                //EM cem = new EM();
                                //em.buildClusterer(data);
                            
				SimpleKMeans skm = new SimpleKMeans(); 
				
				skm.setNumClusters(k);
				
				skm.setDistanceFunction(new EuclideanDistance());
				//skm.setDistanceFunction(new CosineDistance());

				skm.setSeed(new Random().nextInt());
				skm.setPreserveInstancesOrder(true);

				skm.buildClusterer(data);

				SimplifiedSilhouette ss = new SimplifiedSilhouette();
				DistanceFunction df = new EuclideanDistance(data);
				//DistanceFunction df = new CosineDistance(data);

				ssQuality = ss.quality(skm, data, df);

				if (ssQuality > ssBestQuality){
					ssBestQuality = ssQuality;
					clusters = skm.getClusterCentroids();
					bestModel = new SimpleKMeans();
					bestModel = skm;
				}
			}

			if (bestModel != null){
				for (int j = 0; j < data.numInstances(); j++) {
					cluLabels[j] = bestModel.clusterInstance(data.instance(j));
				}
			}
			//System.out.println("-> Clusterer thread " + Thread.currentThread().getId() + " finished! (k=" + k + ")");
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}  
	}

	public double getSSBestQuality() {
		return ssBestQuality;
	}

	public int getK() {
		return k;
	}

	public int[] getCluLabels() {
		return cluLabels;
	}

	public SimpleKMeans getBestModel() {
		return bestModel;
	}

	public Instances getClusters() {
		return clusters;
	}
}
