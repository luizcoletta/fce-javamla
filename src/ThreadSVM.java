import weka.core.Instances;
import weka.classifiers.Evaluation;
//import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.RBFKernel;

public class ThreadSVM extends Thread {
//public class SVMThread implements Runnable {
	
	private SMO svm = null;
	//private LibSVM svm = null;
	private Instances trainData = null;
	private Instances testData = null;
	private double cost = -1;
	private double gamma = -1;
	private String properties = null;
	private double accuracy = -1;

	public ThreadSVM(Instances trainData, Instances testData, double cost, double gamma, String properties) {
		//svm = new LibSVM();
		svm = new SMO();
		this.trainData = trainData;
		this.testData = testData;
		this.cost = cost;
		this.gamma = gamma;
		this.properties = properties;
	}

	public void run() {
		
		//System.out.println("-> Running SVM thread " + Thread.currentThread().getId() + "(Cost=" + cost + " Gamma=" + gamma + ")");
		
		try {
			
			if (!properties.isEmpty()){
				svm.setOptions(weka.core.Utils.splitOptions(properties));
			}
			//svm.setSeed(1);
			//svm.setCost(cost);
			//svm.setGamma(gamma);
			//svm.setProbabilityEstimates(true);
			
			RBFKernel rbf = new RBFKernel();
			rbf.setGamma(gamma);
			
			svm.setKernel(rbf);
			svm.setC(cost);
			
			svm.buildClassifier(trainData);
			
			Evaluation eval = new Evaluation(trainData);
			eval.evaluateModel(svm, testData);
			//System.out.println(eval.toSummaryString("\nResults\n======\n", false));
			accuracy = (eval.correct()*100)/(eval.correct()+eval.incorrect());
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		//System.out.println("-> SVM thread " + Thread.currentThread().getId() + " finished!");
	}

	public double getAccuracy() {
		return accuracy;
	}
	
	public double getCost() {
		return cost;
	}
	
	public double getGamma() {
		return gamma;
	}
}