import weka.core.Instances;
import weka.classifiers.Classifier;

public class ClassifierThread extends Thread {
	
	private Classifier cla = null;
	private Instances trainData = null;

	public ClassifierThread(Classifier cla, Instances trainData) {
		this.cla = cla;
		this.trainData = trainData;
	}

	public void run() {
		
		// System.out.println("-> Running classifier thread " + Thread.currentThread().getId());
		
		try {
			cla.buildClassifier(trainData);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		// System.out.println("-> Classifier thread " + Thread.currentThread().getId() + " finished!");
	}

	public Classifier getCla() {
		return cla;
	}
}