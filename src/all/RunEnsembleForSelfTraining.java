package all;

import weka.classifiers.*;
import weka.classifiers.bayes.NaiveBayes;
//import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.trees.J48;
import weka.classifiers.lazy.IBk;
import weka.clusterers.SimpleKMeans;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.RemoveWithValues;
//import wlsvm.WLSVM;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Random;

import javax.swing.plaf.metal.MetalIconFactory.TreeFolderIcon;

/**
 * Author: 	Luiz F. S. Coletta 
 * Email: 	luiz.fersc@gmail.com
 * Date: 	24/09/2014 
 * Update:	24/09/2014
 */
public class RunEnsembleForSelfTraining {

	private static Instances iTrain = null;
	private static Instances iTest = null;
	private static String sFCAE = "";
	private static String sFCUE = "";
	private static String nameData = "";

	public RunEnsembleForSelfTraining(){

	}

	public static Instances getITrain() {
		return iTrain;
	}

	public static void setITrain(Instances iTrain) {
		RunEnsembleForSelfTraining.iTrain = iTrain;
	}

	public static Instances getITest() {
		return iTest;
	}

	public static void setITest(Instances iTest) {
		RunEnsembleForSelfTraining.iTest = iTest;
	}

	public static String getSFCAE() {
		return sFCAE;
	}

	public static void setSFCAE(String sFCAE) {
		RunEnsembleForSelfTraining.sFCAE = sFCAE;
	}

	public static String getSFCUE() {
		return sFCUE;
	}

	public static void setSFCUE(String sFCUE) {
		RunEnsembleForSelfTraining.sFCUE = sFCUE;
	}

	public static String getNameData() {
		return nameData;
	}

	public static void setNameData(String nameData) {
		RunEnsembleForSelfTraining.nameData = nameData;
	}

	//*************************************************************************
	// Main
	//*************************************************************************
	public static void main (String args[])
	{   

		String data = "/home/luiz/datasets/coil3.arff"; //101objcat64 //corel1000
		String path_results = "results/";  

		int sizeValSet = 100;   // the number of objects in the validation set (keeping stratification)
		int fold = 1;	       // current fold (if fold = 0 it is used only one fold (train and test are the same set)) 

		// CLASSIFIER ENSEMBLE
		ArrayList<Integer> typeClaEns = new ArrayList<Integer>();
		typeClaEns.add(0); // NB
		typeClaEns.add(0); // J48
		typeClaEns.add(0); // KNN
		typeClaEns.add(1); // SVM

		// CLUSTER ENSEMBLE
		// 1: strategy 1 (old)
		// 2: strategy 2 (new)
		int strategyCluEns = 3;

		// number of partitions (1, 2, 3, ...)
		int typeCluEns = 5; 

		// multiplier (for strategy 1) 
		// size of subset of features (for strategy 2)
		int theta = 2; 

		//  0: the train set will contain all classes
		// >0: the train set will not contain the class of this index
		ArrayList<Integer> missingClasses = new ArrayList<Integer>(); 
		missingClasses.add(0); // class index

		//  0: for testing (to build the test and train sets); 
		//  1: for validation (fold 1 of 2 from labeled objects - the dataset's name appears with "R");
		//  2: for validation (fold 2 of 2 from labeled objects - the dataset's name appears with "R");
		int trainTest = 1;

		//  <0: there are no new classes
		// >=0: there are new classes
		ArrayList<Integer> newInst = new ArrayList<Integer>(); 
		newInst.add(-1); // index (starts in 1)

		int printResults = 1;

		String iter = "0";

		GenerateC3EInputs(data, 
				path_results, 
				fold, 
				sizeValSet, 
				typeClaEns, 
				strategyCluEns, 
				theta, 
				typeCluEns, 
				missingClasses,
				trainTest,
				newInst,
				printResults, 
				iter);
	}

	//*********************************************************************************
	// Creating files labels.dat, piSet.dat, and SSet.dat for Semi-supervised Learning
	//*********************************************************************************
	public static ArrayList<Double> GenerateC3EInputs(String data, String path_results, int fold, int sizeValSet, ArrayList<Integer> typeClaEns, int strategyCluEns, int theta, int typeCluEns, ArrayList<Integer> mClasses, int trainTest, ArrayList<Integer> newInst, int printResults, String iter)
	{
		ArrayList<Double> result = new ArrayList<Double>();
		int Tfold = 0;

		try {

			// TRABALHAR AQUI PARA INSERIR TEST SEPARADO QUANDO FOR A ULTIMA ITERACAO DO ALGORITMO
			
			GenerateDataSources(data, 
					path_results, 
					fold, 
					sizeValSet, 
					typeClaEns, 
					strategyCluEns, 
					theta, 
					typeCluEns, 
					mClasses,
					trainTest,
					newInst,
					printResults, 
					iter);

			result = RunEnsembles(getITrain(), 
					getITest(), 
					path_results, 
					getSFCAE(), 
					getSFCUE(), 
					getNameData(), 
					fold, 
					trainTest, 
					typeClaEns, 
					strategyCluEns, 
					theta, 
					typeCluEns, 
					printResults, 
					iter);

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ArrayIndexOutOfBoundsException e) {
			if (fold > Tfold){
				System.out.print("\n-> ERROR!! Number of fold " + fold + " exceeds the limit of " + Tfold + "\n");
			}
			//e.printStackTrace();
		} catch (Exception e) {
			e.printStackTrace();
		} finally{}

		return result;
	}

	//*********************************************************************************
	// Provide Data Sources to be Used by Ensembles
	//*********************************************************************************
	public static ArrayList<Double> GenerateDataSources(String data, String path_results, int fold, int sizeValSet, ArrayList<Integer> typeClaEns, int strategyCluEns, int theta, int typeCluEns, ArrayList<Integer> mClasses, int trainTest, ArrayList<Integer> newInst, int printResults, String iter) throws Exception
	{
		ArrayList<Double> result = new ArrayList<Double>();

		String strmc = "";
		String strni = "0";
		boolean logFile = true;
		//boolean changeSets = false; //just for tweet classification
		boolean changeSets = true;
		double txSizeValSet1 = 0.5;
		double txSizeValSet2 = 0.5;
		int Tfold = 0;

		if (printResults > 0 && changeSets) {System.out.print("\n-> CREATING FILES (labels, piSet and SSet) - train and test sets were switched\n");}
		if (printResults > 0 && !changeSets) {System.out.print("\n-> CREATING FILES (labels, piSet and SSet)\n");}

		// ------------------------------------------------------------
		// BUILDING THE TRAIN AND TEST SETS
		// ------------------------------------------------------------

		// if fold = 0 it is used only one fold (same train and test sets), otherwise is used cross-validation  
		DataSource oneFold = new DataSource(data);
		Instances iTrain = oneFold.getDataSet();
		Instances iTest = oneFold.getDataSet();
		Instances iNInst = oneFold.getDataSet();
		iNInst.delete();

		// to obtain the correct number of folds for the required number of objects in the validation set
		Tfold = (int) Math.ceil(iTrain.numInstances()/sizeValSet);

		if (fold != 0){
			CrossValidationFolds cvf = new CrossValidationFolds(data, Tfold, 0);

			// getting the train set
			if (changeSets){
				iTrain = cvf.getTestData(fold-1); // switched to have less objects in train set than in test set (iTrain = cvf.getTrainData(fold-1);)
			}else{
				iTrain = cvf.getTrainData(fold-1);
			}

			// removing classes from the train set
			for (int i = 0; i < mClasses.size(); i++) {
				if (mClasses.get(i) == 0){
					strmc = "0";
					break;
				} else {

					strmc = strmc + mClasses.get(i).toString();

					RemoveWithValues filter = new RemoveWithValues();

					filter.setAttributeIndex(String.valueOf(iTrain.classIndex()+1));
					filter.setNominalIndices(String.valueOf(mClasses.get(i)));
					filter.setInputFormat(iTrain);
					Instances filteredData = Filter.useFilter(iTrain, filter); 
					iTrain = filteredData;
				}
			}

			// getting the test set
			if (changeSets){
				iTest = cvf.getTrainData(fold-1); // switched to have more objects in test set than in train set (iTest = cvf.getTestData(fold-1);)
			}else{
				iTest = cvf.getTestData(fold-1);
			}

			// inserting new instances in the train set
			if (newInst.get(0) >= 0){
				strni = "1";
				for (int i = 0; i < newInst.size(); i++) {
					iTrain.add(iTest.instance(newInst.get(i)-1));
					iNInst.add(iTest.instance(newInst.get(i)-1));
				}
				for (int i = 0; i < newInst.size(); i++) {
					iTest.delete(newInst.get(i)-1);
				}
			}

			// if validation
			if (trainTest > 0){
				// to obtain the correct number of folds for the required number of objects in the validation set
				int TfoldVS = -1;
				if (sizeValSet < 10){
					TfoldVS = (int) Math.ceil(iTrain.numInstances()/(Math.round(iTrain.numInstances()*txSizeValSet1)-1));
				}
				if (sizeValSet >= 10){
					TfoldVS = (int) Math.ceil(iTrain.numInstances()/(Math.round(iTrain.numInstances()*txSizeValSet2)-1));
				}
				CrossValidationFolds cvft = new CrossValidationFolds(iTrain, TfoldVS, 0);

				if (changeSets){
					iTrain = cvft.getTestData(trainTest-1); 
					iTest = cvft.getTrainData(trainTest-1);
				}else{
					iTrain = cvft.getTrainData(trainTest-1); 
					iTest = cvft.getTestData(trainTest-1);						
				}
			}
		}
		iTrain.setClassIndex(iTrain.numAttributes()-1);
		iTest.setClassIndex(iTrain.numAttributes()-1);

		// ------------------------------------------------------------
		// OBTAINING THE NAMES OF FILES
		// ------------------------------------------------------------			
		String nameData = oneFold.getDataSet().relationName().toLowerCase();
		String sFCAE = "";
		String sFCUE = "";
		if (trainTest > 0){
			sFCAE = "_" + nameData + "R" + fold + strmc + strni + iter + trainTest + typeClaEns.get(0) + typeClaEns.get(1) + typeClaEns.get(2) + typeClaEns.get(3);
			sFCUE = "_" + nameData + "R" + fold + strmc + strni + iter + trainTest + strategyCluEns + theta + typeCluEns;		
		}else{
			sFCAE = "_" + nameData + fold + strmc + strni + iter + typeClaEns.get(0) + typeClaEns.get(1) + typeClaEns.get(2) + typeClaEns.get(3);
			sFCUE = "_" + nameData + fold + strmc + strni + iter + strategyCluEns + theta + typeCluEns;
		}

		// ------------------------------------------------------------
		// CREATING LOG FILES
		// ------------------------------------------------------------		
		if (logFile){

			(new File("files_" + nameData)).mkdirs();

			ArffSaver saverTrain = new ArffSaver();
			saverTrain.setInstances(iTrain);
			saverTrain.setFile(new File("files_" + nameData + "/iTrain" + strni + ".arff"));
			saverTrain.writeBatch();

			ArffSaver saverTest1 = new ArffSaver();
			saverTest1.setInstances(iTest);
			saverTest1.setFile(new File("files_" + nameData + "/iTest" + strni + ".arff"));
			saverTest1.writeBatch();

			ArffSaver saverNewInst = new ArffSaver();
			saverNewInst.setInstances(iNInst);
			saverNewInst.setFile(new File("files_" + nameData + "/iNewInst" + strni + ".arff"));
			saverNewInst.writeBatch();

			ARFFToMatlab.salvar("files_" + nameData + "/iTrain" + strni + ".dat",ARFFToMatlab.carregar("files_" + nameData + "/iTrain" + strni + ".arff"),false);
			ARFFToMatlab.salvar("files_" + nameData + "/iTest" + strni + ".dat",ARFFToMatlab.carregar("files_" + nameData + "/iTest" + strni + ".arff"),false);
			ARFFToMatlab.salvar("files_" + nameData + "/iNewInst" + strni + ".dat",ARFFToMatlab.carregar("files_" + nameData + "/iNewInst" + strni + ".arff"),false);

			if ((trainTest == 0) && (strni.equals("1"))){
				ARFFToMatlab.salvar("files_" + nameData + "/labeledObjects.dat",ARFFToMatlab.carregar("files_" + nameData + "/iNewInst" + strni + ".arff"),false);
			}
			if (strategyCluEns == 3){ ARFFToMatlab.salvar("clustering/iTest.dat",ARFFToMatlab.carregar("files_" + nameData + "/iTest" + strni + ".arff"),false); }
		}

		setITrain(iTrain);
		setITest(iTest);
		setNameData(nameData);
		setSFCAE(sFCAE);
		setSFCUE(sFCUE);

		AnalysingMemory(printResults);

		return result;
	}

	//******************************************************************************
	// Run Supervised and Unsupervised Ensembles for Semi-supervised Scenarios
	//******************************************************************************
	public static ArrayList<Double> RunEnsembles(Instances iTrain, Instances iTest, String path_results, String sFCAE, String sFCUE, String nameData, int fold, int trainTest, ArrayList<Integer> typeClaEns, int strategyCluEns, int theta, int typeCluEns, int printResults, String iter) throws Exception
	{
		ArrayList<Double> result = new ArrayList<Double>();
		boolean logFile = true;
		FileWriter writer = null;
		PrintWriter out = null;

		// ----------------------------------------------------------------
		// RUNNING SUPERVISED MODELS (sum of the typeClaEns vector > 0)
		// ----------------------------------------------------------------
		int runClassEns = 0;
		runClassEns = typeClaEns.get(0) + typeClaEns.get(1) + typeClaEns.get(2) + typeClaEns.get(3);
		if (runClassEns > 0){

			// ---------------------------------------------------
			// save labels of objects in file labels.dat
			// ---------------------------------------------------
			FileWriter wLabels = new FileWriter(new File(path_results + "labels" + sFCAE + ".dat"),false);
			PrintWriter outLabels = new PrintWriter(wLabels);
			for (int i = 0; i < iTest.numInstances(); i++) {
				double pc = iTest.instance(i).value(iTest.classIndex())+1;
				outLabels.println(pc);
			}
			outLabels.close();
			wLabels.close();

			// ---------------------------------------------------
			// load classifiers
			// ---------------------------------------------------
			int contCla = 0;
			Classifier[] vcls = new Classifier[runClassEns];
			ArrayList<String> properties = LoadEnsembleSettings();
			if (printResults > 0) {System.out.print("-> Building the classifier ensemble\n");}
			if (typeClaEns.get(0) == 1){

				NaiveBayes nb = new NaiveBayes();

				if (!properties.get(0).isEmpty()){
					nb.setOptions(weka.core.Utils.splitOptions(properties.get(0)));
				}

				vcls[contCla] = nb;

				if (printResults > 0) {System.out.print("\n-> Naive Bayes [" + properties.get(0) + "]");}
				contCla++;
			}
			if (typeClaEns.get(1) == 1){

				J48 j48 = new J48();

				if (!properties.get(1).isEmpty()){
					j48.setOptions(weka.core.Utils.splitOptions(properties.get(1)));
				}
				j48.setUseLaplace(true);

				vcls[contCla] = j48; 

				if (printResults > 0) {System.out.print("\n-> J48 [" + properties.get(1) + "]");}
				contCla++;
			}
			if (typeClaEns.get(2) == 1){

				IBk knn = new IBk();

				if (!properties.get(2).isEmpty()){
					knn.setOptions(weka.core.Utils.splitOptions(properties.get(2)));
				}

				vcls[contCla] = knn;

				if (printResults > 0) {System.out.print("\n-> KNN [" + properties.get(2) + "]");}
				contCla++;
			}      
			if (typeClaEns.get(3) == 1){

				//LibSVM svm = new LibSVM();
				SMO svm = new SMO();

				if (!properties.get(3).isEmpty()){
					svm.setOptions(weka.core.Utils.splitOptions(properties.get(3)));
				}

				// ---------------------------------------------------
				// grid search (parameters cost and gamma)
				// ---------------------------------------------------
				boolean gridSearchSVM = true;
				if (gridSearchSVM){

					Instances iTrainSVM = iTrain;
					Instances iTestSVM = iTest;

					if (fold != 0){
						if (trainTest == 0){
							// to obtain the correct number of folds for the required number of objects in the validation set
							int TfoldVS = -1;
							if (iTrain.numInstances() < 10){
								TfoldVS = (int) Math.ceil(iTrainSVM.numInstances()/Math.round(iTrain.numInstances()*0.5));
							}
							if (iTrain.numInstances() >= 10){
								TfoldVS = (int) Math.ceil(iTrainSVM.numInstances()/Math.round(iTrain.numInstances()*0.3));
							}
							CrossValidationFolds cvft = new CrossValidationFolds(iTrainSVM, TfoldVS, 0);
							iTrainSVM = cvft.getTestData(0); 
							iTestSVM = cvft.getTrainData(0);
						}
					}

					GridSearchThreadSVM svmGS = new GridSearchThreadSVM(iTrainSVM, iTestSVM, properties.get(3));
					svmGS.runGridSearch();
					double bestCost = svmGS.getBestCost();
					double bestGamma = svmGS.getBestGamma();

					if (logFile){
						StringBuffer buf = new StringBuffer();
						buf.append("Cost=" + bestCost + "\t Gamma=" + bestGamma + "\t Acc=" + svmGS.getBestFit() + "\n");
						ARFFToMatlab.salvar("files_" + nameData + "/SVM_Grid-Search.dat", buf, true);
					}

					//svm.setCost(bestCost); 
					//svm.setGamma(bestGamma); 

					RBFKernel rbf = new RBFKernel();
					rbf.setGamma(bestGamma);

					svm.setKernel(rbf);
					svm.setC(bestCost); 
				}		

				//svm.setSeed(1);
				//svm.setProbabilityEstimates(true);

				vcls[contCla] = svm;

				if (printResults > 0) {System.out.print("\n-> SVM [" + properties.get(3) + "]");}
			}  
			if (printResults > 0) {System.out.print("\n\n");}

			// ---------------------------------------------------
			// RUNNING CLASSIFIER ENSEMBLE 
			// ---------------------------------------------------
			ArrayList<double[]> probClass = new ArrayList<double[]>();
			ClassifierEnsembleThread cla = new ClassifierEnsembleThread(vcls, iTrain, iTest);
			cla.runTrain();
			probClass = cla.runTest(); // class probability distribution (piSet)

			// saving file piSet.dat
			writer = new FileWriter(new File(path_results + "piSet" + sFCAE + ".dat"),false);
			out = new PrintWriter(writer);
			for (int i = 0; i < probClass.size(); i++) {
				double[] pc = probClass.get(i);
				for (int j = 0; j < pc.length; j++) {
					out.print(pc[j] + "\t");
				}
				out.println();
			}
			out.close();
			writer.close();
		}else{
			if (printResults > 0) {System.out.println("-> Did not build the classifier ensemble!");}
		}

		AnalysingMemory(printResults);

		// ----------------------------------------------------------------
		// RUNNING UNSUPERVISED MODELS (typeCluEns > 0)
		// ----------------------------------------------------------------
		if (typeCluEns > 0){

			if (printResults > 0) {System.out.print("-> Building the cluster ensemble (strategy " + strategyCluEns + ")\n");}   

			// ---------------------------------------------------
			// remove class attribute
			// ---------------------------------------------------
			Remove remove;
			remove = new Remove();
			remove.setAttributeIndices(String.valueOf(iTest.numAttributes()));
			remove.setInvertSelection(new Boolean(false).booleanValue());
			remove.setInputFormat(iTest);
			iTest = Filter.useFilter(iTest, remove);

			ArrayList<Integer> kMin = new ArrayList<Integer>();
			ArrayList<Integer> kMax = new ArrayList<Integer>();
			ArrayList<int[]> subAtt = null;

			if (strategyCluEns == 1){

				// ---------------------------------------------------
				// optimization of parameters (k)
				// ---------------------------------------------------
				int bestK = 0;
				double bestQuality = Double.NEGATIVE_INFINITY;

				ArrayList<Integer> nClusters = new ArrayList<Integer>();

				//initial and final k come from the number of classes and instances respectively.
				for (int i = iTrain.numClasses(); i<=Math.round(Math.sqrt(iTest.numInstances())); ++i){nClusters.add(i);}					

				ClustererOMRThread cluOMR = new ClustererOMRThread(nClusters, 20, iTest);
				bestK = cluOMR.Run();
				bestQuality = cluOMR.getSSBestQuality();

				if (printResults > 0) {System.out.print("\n-> Using strategy 1\n");}
				if (printResults > 0) {System.out.print("\n-> Cluster Ensemble - Parameter Optimization (k):\n");}
				if (printResults > 0) {System.out.print("\n-> Best Simplified Silhouette: " + bestQuality + "\n");}
				if (printResults > 0) {System.out.print("-> Best Number of Clusters: " + bestK + "\n");}

				// ---------------------------------------------------
				// setting unsupervised model up
				// ---------------------------------------------------
				if (typeCluEns == 1){
					kMin.add(bestK*theta);
					kMax.add(bestK*theta);
				}else if (typeCluEns == 2){
					kMin.add(bestK*theta);
					kMin.add(bestK*2*theta);
					kMax.add(bestK*theta);
					kMax.add(bestK*2*theta);
				}else if (typeCluEns == 3){
					kMin.add(bestK*theta);
					kMin.add(bestK*2*theta);
					kMin.add(bestK*3*theta);
					kMax.add(bestK*theta);
					kMax.add(bestK*2*theta);
					kMax.add(bestK*3*theta);
				}
			}

			if (strategyCluEns == 2){

				int varKMin = iTrain.numClasses()*2;
				int varKMax = (int)Math.round(Math.sqrt(iTest.numInstances()));

				if (varKMax < 2){
					varKMax = 2;
				}
				if (varKMin > varKMax){
					varKMin = varKMax;
				}

				for (int i = 1; i<=typeCluEns; ++i){kMin.add(varKMin);}	
				for (int i = 1; i<=typeCluEns; ++i){kMax.add(varKMax);}	

				//generating subsets of attributes 
				boolean subSetAtt = true;
				double[] pArray = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1};
				double pAtt = pArray[theta-1]; //using variable theta to get size of the subset

				if ((subSetAtt) && (iTest.numAttributes()>2) && ((iTest.numAttributes()*pAtt)*typeCluEns<=iTest.numAttributes())){
					int maxAtt = iTest.numAttributes();
					int maxSubAtt = (int) Math.round(maxAtt*pAtt);
					int iAtt[] = new int[maxAtt];
					subAtt = new ArrayList<int[]>();

					for (int i = 0; i<maxAtt; ++i){iAtt[i] = i;}	
					for (int j = 1; j<=typeCluEns; ++j){
						int iSubAtt[] = new int[maxSubAtt];
						for (int i = 0; i<maxSubAtt; ++i){
							boolean inserted = true;
							while (inserted){
								int rnd1 = RandomAttInd(maxAtt);
								iSubAtt[i] = rnd1;
								inserted = false;
								if (iAtt[rnd1] == -1){
									inserted = true;
								}
								iAtt[rnd1] = -1;
							}
						}
						subAtt.add(iSubAtt);
					}
				}
			}

			// Using ELKI package to cluster objects from iTest
			if (strategyCluEns == 3){

				String nFile = "clustering/iTest.dat";

				File fileTest = new File(nFile);
				if (fileTest.exists()) {

					int[] kArray = {2,3,5,6,8};
					//int[] kArray = {5,10,15,20,25};
					for (int i = 0; i < typeCluEns; i++) {

						String k = String.valueOf(kArray[i]);

						String cluster = String.valueOf(i+1);

						String runELKI1 = "java -jar clustering/elki.jar KDDCLIApplication " +
								"-dbc.in clustering/iTest.dat " +
								"-algorithm clustering.kmeans.KMedoidsPAM " +
								"-algorithm.distancefunction CosineDistanceFunction " +
								"-kmeans.k ";

						String runELKI2 = " -kmeans.initialization RandomlyChosenInitialMeans " +
								"-resulthandler ResultWriter -out clustering/";

						String runELKI3 = new StringBuilder(runELKI1).append(k).append(runELKI2).append(cluster).toString();

						Process proc = Runtime.getRuntime().exec(runELKI3);
						proc.waitFor();
					}
				}
			}

			AnalysingMemory(printResults);

			// ---------------------------------------------------
			// RUNNING CLUSTER ENSEMBLE
			// ---------------------------------------------------
			if (strategyCluEns != 3){
				ClustererEnsemble clue = new ClustererEnsemble(kMin, kMax, 20, iTest, subAtt);
				clue.buildCluster();
				float caMatrix[][] = clue.createCMatrix(); // creating similarity (co-association) matrix (SSet)

				// saving file SSet.dat
				writer = new FileWriter(new File(path_results + "SSet" + sFCUE + ".dat"),false);
				out = new PrintWriter(writer);
				for (int i = 0; i < caMatrix.length ; i++) {
					for (int j = 0; j < caMatrix[0].length; j++) {
						out.print(caMatrix[i][j] + "\t");
					}
					out.println();
				}
				out.close();
				writer.close();
			}else{	

				File file = new File("clustering");
				if (file.exists()) {

					ObtainCoAssMatrix oCAM = new ObtainCoAssMatrix();

					float caMatrix[][] = oCAM.createCMatrix();

					// saving file SSet.dat
					writer = new FileWriter(new File(path_results + "SSet" + sFCUE + ".dat"),false);
					out = new PrintWriter(writer);
					for (int i = 0; i < caMatrix.length ; i++) {
						for (int j = 0; j < caMatrix[0].length; j++) {
							out.print(caMatrix[i][j] + "\t");
						}
						out.println();
					}
					out.close();
					writer.close();
				} else {
					if (printResults > 0) {System.out.println("-> Did not build the cluster ensemble!");}
				}			
			}

		}else{
			if (printResults > 0) {System.out.println("-> Did not build the cluster ensemble!");}
		}

		AnalysingMemory(printResults);

		if (printResults > 0) {System.out.print("-> Hoooray! This is the end my friend!\n\n");}

		return result;
	}


	private static void AnalysingMemory(int printResults){

		final long MEGABYTE = 1024L * 1024L;
		Runtime runtime = Runtime.getRuntime(); // get the java runtime
		long memory = runtime.totalMemory() - runtime.freeMemory(); // calculate the used memory
		long maxMemory = runtime.maxMemory();   // max heap memory

		runtime.gc();

		if (printResults > 0) {System.out.println("\n-> Used memory (megabytes): " + memory/MEGABYTE);}
		if (printResults > 0) {System.out.println("-> Heap memory (megabytes): " + maxMemory/MEGABYTE + "\n");}
	}

	private static int RandomAttInd(int maxAtt){
		return 0 + (int)(Math.random()*((maxAtt-0)+0));
	}


	private static ArrayList<String> LoadEnsembleSettings(){

		BufferedReader br = null;
		ArrayList<String> properties = new ArrayList<String>();
		properties.add("");
		properties.add("");
		properties.add("");
		properties.add("");

		File f = null;

		try{
			//String currentPath = new File(".").getCanonicalPath();
			ClassLoader loader = RunEnsembleForSelfTraining.class.getClassLoader();
			String currentPath = loader.getResource("").getFile();

			f = new File(currentPath + "settings.dat");
			if(f.exists()){
				br = new BufferedReader(new FileReader(currentPath + "settings.dat"));
			}

			String line = null;
			while((line = br.readLine()) != null ){

				String[] LV = line.split("\t");

				if (!LV[0].isEmpty()){
					if (!LV[0].substring(0,1).equals("#")){
						if (LV[0].length() > LV[0].indexOf("]")+2){
							if (LV[0].substring(1, LV[0].indexOf("]")).equals("NB")){
								properties.set(0, LV[0].substring(LV[0].indexOf("]")+2,LV[0].length()));
								//System.out.println(properties.get(0));
							}
							if (LV[0].substring(1, LV[0].indexOf("]")).equals("J48")){
								properties.set(1, LV[0].substring(LV[0].indexOf("]")+2,LV[0].length()));
								//System.out.println(properties.get(1));
							}
							if (LV[0].substring(1, LV[0].indexOf("]")).equals("KNN")){
								properties.set(2, LV[0].substring(LV[0].indexOf("]")+2,LV[0].length()));
								//System.out.println(properties.get(2));
							}
							if (LV[0].substring(1, LV[0].indexOf("]")).equals("SVM")){
								properties.set(3, LV[0].substring(LV[0].indexOf("]")+2,LV[0].length()));
								//System.out.println(properties.get(3));
							}
						}
					}
				}
			}
			br.close();
		}catch(FileNotFoundException e){
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}

		return properties;
	}

}