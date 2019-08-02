import java.util.ArrayList;
import weka.classifiers.Classifier;
import weka.core.Instances;

public class ClassifierEnsembleThread {

	private Instances trainData = null;
	private Instances testData = null;
	private Classifier[] MClassifiers = null;
	
	public ClassifierEnsembleThread (Classifier[] m_Classifiers, Instances trainData, Instances testData) {
		this.trainData = trainData;
		this.testData = testData;
		this.MClassifiers = m_Classifiers;
	}

	public void runTrain() throws InterruptedException {
		
		ArrayList<ClassifierThread> threads = new ArrayList<ClassifierThread>();  
		
		for (Classifier c : MClassifiers){
			ClassifierThread task = new ClassifierThread(c, trainData);  
			threads.add(task); 
		}

		// start threads  
		for (ClassifierThread t : threads){  
			t.start();  
		}  
		for (ClassifierThread t : threads){  
			t.join();  
		}  

		// getting results
		// System.out.println("\n-> Getting classifiers results");
		int i = 0;
		for (ClassifierThread t : threads){  
			//System.out.println("-> k = " + t.getK());
			MClassifiers[i] = t.getCla();
			i++;
		}  
	}

	public ArrayList<double[]> runTest() throws Exception {

		ArrayList<double[]> probClass = new ArrayList<double[]>();

		for (int i = 0; i < testData.numInstances(); i++) {

			double[] probClassI = new double[testData.numClasses()];

			for (int j = 0; j < MClassifiers.length; j++) {

				double[] pci = MClassifiers[j].distributionForInstance(testData.instance(i));

				for (int k = 0; k < testData.numClasses(); k++) {
					probClassI[k] += pci[k];
					if (j == MClassifiers.length-1) {
						probClassI[k] = probClassI[k] / MClassifiers.length;
					}
				}

			}
			probClass.add(probClassI);
		}
		return probClass;
	}
}
