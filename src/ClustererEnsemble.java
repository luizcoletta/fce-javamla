import java.util.ArrayList;

import weka.clusterers.Clusterer;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 * Author: Luiz F. S. Coletta 
 * Email: luiz.fersc@gmail.com
 * Date: 15/10/2013 
 */
public class ClustererEnsemble {

	private Clusterer[] cluEnsemble = null;
	private ArrayList<Integer> kMin = null;
	private ArrayList<Integer> kMax = null;
	private Instances data = null;
	private int nInit = 50;
	private ArrayList<int[]> subAtt = null;;
	
	private ClustererEnsembleThread cluEns = null;

	public ClustererEnsemble (ArrayList<Integer> kMin, ArrayList<Integer> kMax, int nInit, Instances data, ArrayList<int[]> subAtt) {
		this.cluEnsemble = new Clusterer[kMin.size()];
		this.kMin = kMin;
		this.kMax = kMax;
		this.nInit = nInit;
		this.data = data;
		this.subAtt = subAtt;
	}

	public void buildCluster() throws Exception {
		cluEns = new ClustererEnsembleThread(kMin, kMax, nInit, data, subAtt);
		cluEns.Run();
		cluEnsemble = cluEns.getCluEnsemble();
	}

	public float[][] createCMatrix() throws Exception {

		float caMatrix[][] = new float [data.numInstances()][data.numInstances()];

		for (int i = 0; i < cluEnsemble.length; i++) {
			for (int j = 0; j < data.numInstances(); j++) {
				for (int k = 0; k < data.numInstances(); k++) {
					if (cluEns.getCluLabel(i,j) == cluEns.getCluLabel(i,k)){
						caMatrix[j][k] += 1;
					}
					if (i==(cluEnsemble.length-1)){
						caMatrix[j][k] = caMatrix[j][k]/cluEnsemble.length;
					}
				}
			}
		}
		
		return caMatrix;
	}
}
