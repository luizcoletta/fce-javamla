import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

import weka.clusterers.Clusterer;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class ClustererEnsembleThread {

	private ArrayList<Integer> kMin = null;
	private ArrayList<Integer> kMax = null;
	private int nInit = 50;
	private Instances data = null;
	private ArrayList<int[]> subAtt = new ArrayList<int[]>();
	private ArrayList<int[]> cluLabels = new ArrayList<int[]>();
	private Clusterer[] cluEnsemble = null;

	public ClustererEnsembleThread(ArrayList<Integer> kMin, ArrayList<Integer> kMax, int nInit, Instances data, ArrayList<int[]> subAtt){
		this.cluEnsemble = new Clusterer[kMin.size()];
		this.kMin = kMin;
		this.kMax = kMax;
		this.nInit = nInit;
		this.data = data;
		this.subAtt = subAtt;
	}

	public void Run() throws InterruptedException{  

		ArrayList<ClustererOMREnsThread> lCluOMREns = new ArrayList<ClustererOMREnsThread>();  

		int it = -1;
		Instances dt = null;
		String infoAtt = null;
		
		for (int j=0; j<kMin.size(); j++){

			ArrayList<Integer> varK = new ArrayList<Integer>();
			ClustererOMREnsThread cluOMREns = null;
			
			for (int i = kMin.get(j); i<=kMax.get(j); ++i){varK.add(i);}

			if (subAtt != null){
				try {
					Remove remove = new Remove();
					int[] sAtt = subAtt.get(++it);
					remove.setAttributeIndicesArray(sAtt);
					remove.setInvertSelection(new Boolean(true).booleanValue());
					remove.setInputFormat(data);
					dt = Filter.useFilter(data, remove);

					infoAtt = infoAtt + dt.toSummaryString(); 
					
					cluOMREns = new ClustererOMREnsThread(varK, nInit, dt);

				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}else{
	
				cluOMREns = new ClustererOMREnsThread(varK, nInit, data); 
			}
			lCluOMREns.add(cluOMREns); 
		}

		for (ClustererOMREnsThread clu : lCluOMREns){  
			clu.Run();  
		}  

		// getting results
		System.out.println("\n-> Getting clusterers results (" + kMin.size() + " partitions)");
                System.out.println("-> kMin = " + kMin.get(0) + ", kMax = " + kMax.get(0) + ", runs = " + nInit);
                System.out.println("-> Best partitions from Simplified Silhouette:");
		int i = 0;
		for (ClustererOMREnsThread clu : lCluOMREns){  
			System.out.println("-> k = " + clu.getBestK() + ", criterion = " + String.valueOf(clu.getSSBestQuality()));
			cluEnsemble[i] = clu.getBestModel();
			cluLabels.add(clu.getCluLabels());
			i++;
		}  

		if (subAtt != null){
			FileWriter fw;
			try {
				fw = new FileWriter("files_" + data.relationName().substring(0, data.relationName().indexOf('-')) + "/clustererAttributes.dat", false);
				fw.write(infoAtt.toString());
				fw.close();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}

	public Clusterer[] getCluEnsemble() {
		return cluEnsemble;
	}

	public int getCluLabel(int clu, int inst) {
		return cluLabels.get(clu)[inst];
	}
}
