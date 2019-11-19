package all;

import weka.classifiers.*;
import weka.classifiers.bayes.NaiveBayes;
//import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.trees.J48;
import weka.classifiers.lazy.IBk;
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

/**
 * Author: 	Luiz F. S. Coletta 
 * Email: 	luiz.fersc@gmail.com
 * Date: 	13/01/2014 
 * Update:	13/01/2014
 */
public class RunEnsembleCrossValidation {

	//*************************************************************************
	// Main
	//*************************************************************************
	public static void main (String args[]) throws IOException
	{   
		String data = "/mnt/Backup/Storage/Data/lungcancer.arff"; //101objcat64 //corel1000
		String path_results = "results/";  

		int maxFolds = 5;      // the number of objects in the validation set (keeping stratification)
		int fold = 1;	       // current fold (if fold = 0 is used only one fold (train and test are the same set)) 

		// CLASSIFIER ENSEMBLE
		ArrayList<Integer> typeClaEns = new ArrayList<Integer>();
		typeClaEns.add(1); // NB
		typeClaEns.add(1); // J48
		typeClaEns.add(0); // KNN
		typeClaEns.add(0); // SVM

		// CLUSTER ENSEMBLE
		// 1: strategy 1 (old)
		// 2: strategy 2 (new)
		int strategyCluEns = 2;

		// number of partitions (1, 2, 3, ...)
		int typeCluEns = 10; 

		// multiplier (for strategy 1) 
		// size of subset of features (for strategy 2)
		int theta = 1; 

		//  0: the train set will contain all classes
		// >0: the train set will not contain the class of this index
		ArrayList<Integer> missingClasses = new ArrayList<Integer>(); 
		missingClasses.add(0); // class index

		//  0: for testing (to build the test and train sets); 
		//  1: for validation (fold 1 of 2 from labeled objects - the dataset's name appears with "R");
		//  2: for validation (fold 2 of 2 from labeled objects - the dataset's name appears with "R");
		int trainTest = 1;

		//  <0: there are not new classes
		// >=0: there are new classes
		ArrayList<Integer> newInst = new ArrayList<Integer>(); 
		newInst.add(-1); // index;

		int printResults = 1;

		String iter = "1";

		RunEnsembleCV(data, 
				path_results, 
				fold, 
				maxFolds, 
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

	//******************************************************************************
	// Creating files labels.dat, piSet.dat, and SSet.dat - semi-supervised learning
	//******************************************************************************
	public static ArrayList<Double> RunEnsembleCV(String data, String path_results, int fold, int Tfold, ArrayList<Integer> typeClaEns, int strategyCluEns, int theta, int typeCluEns, ArrayList<Integer> mClasses, int trainTest, ArrayList<Integer> newInst, int printResults, String iter)
	{
		ArrayList<Double> result = new ArrayList<Double>();

		String strmc = "";
		String strni = "0";
		boolean logFile = true;

		if (printResults > 0) {System.out.print("\n-> CREATING FILES (labels, piSet and SSet)\n");}

		try {

			FileWriter writer = null;
			PrintWriter out = null;

			// ------------------------------------------------------------
			// BUILDING THE TRAIN AND TEST SETS
			// ------------------------------------------------------------

			// if fold = 0 is used only one fold (same train and test sets), otherwise is used cross-validation  
			DataSource oneFold = new DataSource(data);
			Instances iTrain = oneFold.getDataSet();
			Instances iTest = oneFold.getDataSet();
			Instances iNInst = oneFold.getDataSet();
			iNInst.delete();

			if (fold != 0){
				CrossValidationFolds cvf = new CrossValidationFolds(data, Tfold, 0);

				// getting the train set
				iTrain = cvf.getTestData(fold-1); // switched to have less objects in train set than in test set (iTrain = cvf.getTrainData(fold-1);)

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
				iTest = cvf.getTrainData(fold-1); // switched to have more objects in test set than in train set (iTest = cvf.getTestData(fold-1);)

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
					int TfoldVS = 2;
					CrossValidationFolds cvft = new CrossValidationFolds(iTrain, TfoldVS, 0);
					iTrain = cvft.getTestData(trainTest-1); 
					iTest = cvft.getTrainData(trainTest-1);
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
			}

			AnalysingMemory(printResults);

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
								int TfoldVS = 2;
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
					if (Math.round(Math.sqrt(iTest.numInstances())) < iTrain.numClasses()){
						for (int i = 1; i<=Math.round(Math.sqrt(iTest.numInstances())); ++i){nClusters.add(i);}
					}else{
						for (int i = iTrain.numClasses(); i<=Math.round(Math.sqrt(iTest.numInstances())); ++i){nClusters.add(i);}
					}
										
					ClustererOMRThread cluOMR = new ClustererOMRThread(nClusters, 20, iTest);
					bestK = cluOMR.Run();
					bestQuality = cluOMR.getSSBestQuality();

					if (printResults > 0) {System.out.print("\n-> Using strategy 1\n");}
					if (printResults > 0) {System.out.print("\n-> Cluster Ensemble - Parameter Optimization (k):\n");}
					if (printResults > 0) {System.out.print("\n-> Best Simplified Silhouette: " + bestQuality + "\n");}
					if (printResults > 0) {System.out.print("-> Best Number of Clusters: " + bestK + "\n");}

					// ---------------------------------------------------
					// setting up unsupervised model
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

				AnalysingMemory(printResults);

				// ---------------------------------------------------
				// RUNNING CLUSTER ENSEMBLE
				// ---------------------------------------------------
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
				if (printResults > 0) {System.out.println("-> Did not build the cluster ensemble!");}
			}

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

		AnalysingMemory(printResults);

		if (printResults > 0) {System.out.print("-> Hoooray! This is the end my friend!\n\n");}

		return result;
	}

	private static void AnalysingMemory(int printResults){

		final long MEGABYTE = 1024L * 1024L;
		Runtime runtime = Runtime.getRuntime(); // get the java runtime
		long memory = runtime.totalMemory() - runtime.freeMemory(); // calculate the used memory
		long maxMemory = runtime.maxMemory(); // max heap memory

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
			ClassLoader loader = RunEnsembleCrossValidation.class.getClassLoader();
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