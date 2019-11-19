package all;

import weka.classifiers.*;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.trees.J48;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SimpleLogistic;
import weka.classifiers.trees.RandomForest;

/**
 * Author: 	Luiz F. S. Coletta 
 * Email: 	luiz.fersc@gmail.com
 * Date: 	20/02/2017 
 * Update:	15/05/2018 
 */
public class RunEnsembleSuppliedTestSet {

	//*************************************************************************
	// Main
	//*************************************************************************
	public static void main (String args[])
	{   
		
                //int sizeX = 100000000;
                //float caMatrix[][] = new float [sizeX][sizeX];
                //float caVet[]= new float [sizeX];
            
                String path_results = "results/";  

                //MODIFICAR
		// CLASSIFIER ENSEMBLE
		ArrayList<Integer> typeClaEns = new ArrayList<>();
		typeClaEns.add(0); // NB
		typeClaEns.add(0); // J48
		typeClaEns.add(0); // KNN
		typeClaEns.add(0); // SVM
                //Modificado por Doug: Adição de novos classificadores.
                typeClaEns.add(0); // BayesNet
                typeClaEns.add(0); // Logistic
                typeClaEns.add(0); // Multilayer Perceptron
                typeClaEns.add(0); // SimpleLogistic
                typeClaEns.add(0); // RandomForest

		// CLUSTER ENSEMBLE
		// 1: strategy 1 (old)
		// 2: strategy 2 (new)
                // 3: stragegy based on KMedoids (using ELKI)
		int strategyCluEns = 2;

		// number of partitions (1, 2, 3, ...)
		int numPartitions = 3; 

		// multiplier (for strategy 1) 
		// size of subset of features (for strategy 2)
		int viewSize = 2; 

		//  0: the train set will contain all classes
		// >0: the train set will not contain the class of this index
		ArrayList<Integer> missingClasses; 
                missingClasses = new ArrayList<>();
		missingClasses.add(0); // class index

		//  0: for testing (to build the test and train sets); 
		//  1: for validation (fold 1 of 2 from labeled objects - the dataset's name appears with "R");
		//  2: for validation (fold 2 of 2 from labeled objects - the dataset's name appears with "R");
		int validation = 0;

		//  <0: there are not new instances (to be incorporated)
		// >=0: there are new instances (to be incorporated)
		//ArrayList<Integer> newInst = new ArrayList<Integer>(); 
		//newInst.add(-1); // index;

		int printResults = 1;
                
                String sFCAE = "";
                String sFCUE = "";
                if (validation > 0){
                        sFCAE = "_" + "hcr" + "R" + "0" + "-1" + "1" + typeClaEns.get(0) + typeClaEns.get(1) + typeClaEns.get(2) + typeClaEns.get(3) + typeClaEns.get(4) + typeClaEns.get(5) + typeClaEns.get(6) + typeClaEns.get(7) + typeClaEns.get(8);
                        sFCUE = "_" + "hcr" + "R" + "0" + "-1" + "1" + strategyCluEns + viewSize + numPartitions;		
                }else{
                        sFCAE = "_" + "hcr" + "0" + "-1" + "1" + typeClaEns.get(0) + typeClaEns.get(1) + typeClaEns.get(2) + typeClaEns.get(3) + typeClaEns.get(4) + typeClaEns.get(5) + typeClaEns.get(6) + typeClaEns.get(7) + typeClaEns.get(8);
                        sFCUE = "_" + "hcr" + "0" + "-1" + "1" + strategyCluEns + viewSize + numPartitions;
                }

		String iter = "0";
                  
                //String trainData = "/mnt/Dropbox/Dropbox/MyFiles/Codes/Matlab/DetectionOfPathogens/data/ceratocystis_train_90_10_rgb-30.arff";
                //String testData = "/mnt/Dropbox/Dropbox/MyFiles/Codes/Matlab/DetectionOfPathogens/data/ceratocystis_test_90_10_rgb.arff"; 
                
                String trainData = RunEnsembleSuppliedTestSet.class.getProtectionDomain().getCodeSource().getLocation().getPath() + "hcr_hashlex_testset.arff"; 
		String testData = RunEnsembleSuppliedTestSet.class.getProtectionDomain().getCodeSource().getLocation().getPath() + "hcr_hashlex_trainset.arff"; 

		RunEnsembleSTS(trainData, 
				testData,
				path_results, 
				typeClaEns, 
				strategyCluEns, 
				viewSize, 
				numPartitions, 
				validation,
				printResults, 
				iter,
                                sFCAE,
                                sFCUE);
	}

	//******************************************************************************
	// Creating files labels.dat, piSet.dat, and SSet.dat - sentiment analysis
	//******************************************************************************
	public static ArrayList<String> RunEnsembleSTS(String trainData, String testData, String path_results, ArrayList<Integer> typeClaEns, int strategyCluEns, int viewSize, int numPartitions, int validation, int printResults, String iter, String sFCAE, String sFCUE)
	{
		ArrayList<String> result = new ArrayList<>();

		String strmc = "0";
		String strni = "0";
		boolean logFile = true;
                String nameDir = "";
                
                // *************************************************************
                // These values are settings for specific experiments
                // *************************************************************
                int numFoldsGSSVM = 2;
                

		if (printResults > 0) {System.out.print("\n-> CREATING FILES (labels, piSet and SSet)\n");}

		try {

			FileWriter writer = null;
			PrintWriter out = null;

			// ------------------------------------------------------------
			// BUILDING THE TRAIN AND TEST SETS
			// ------------------------------------------------------------ 
                        if (trainData.equals("")){
                            trainData = RunEnsembleSuppliedTestSet.class.getProtectionDomain().getCodeSource().getLocation().getPath() + "leaves-train.arff";
                        }
                        if (testData.equals("")){
                            testData = RunEnsembleSuppliedTestSet.class.getProtectionDomain().getCodeSource().getLocation().getPath() + "leaves-test.arff";
                        }
			DataSource oneFoldTrain = new DataSource(trainData);
			DataSource oneFoldTest = new DataSource(testData);
			
			Instances iTest = oneFoldTest.getDataSet();
                        Instances iTrain = oneFoldTrain.getDataSet();
                        
                        // if validation
			if (validation > 0){
				CrossValidationFolds cvft = new CrossValidationFolds(iTrain, 2, 0);
				iTrain = cvft.getTestData(validation-1); 
				iTest = cvft.getTrainData(validation-1);
			}

			iTrain.setClassIndex(iTrain.numAttributes()-1);
			iTest.setClassIndex(iTest.numAttributes()-1);

			// ------------------------------------------------------------
			// OBTAINING THE NAMES OF FILES
			// ------------------------------------------------------------			
			String nameData1 = oneFoldTrain.getDataSet().relationName().toLowerCase();
                        String nameData2 = oneFoldTest.getDataSet().relationName().toLowerCase();
                        result.add("labels" + sFCAE + ".dat");
                        result.add("piSet" + sFCAE + ".dat");
                        result.add("SSet" + sFCUE + ".dat");
                        
			// ------------------------------------------------------------
			// CREATING LOG FILES
			// ------------------------------------------------------------		
			if (logFile){
                                
                                nameDir = "files_" + nameData1;
                                if (nameData1.indexOf('-') != -1){
                                    nameDir = "files_" + nameData1.substring(0, nameData1.indexOf('-'));
                                }
                                
                                (new File(nameDir)).mkdirs();
				
				ArffSaver saverTrain = new ArffSaver();
				saverTrain.setInstances(iTrain);
				saverTrain.setFile(new File(nameDir + "/iTrain" + strni + ".arff"));
				saverTrain.writeBatch();

				ArffSaver saverTest1 = new ArffSaver();
				saverTest1.setInstances(iTest);
				saverTest1.setFile(new File(nameDir + "/iTest" + strni + ".arff"));
				saverTest1.writeBatch();

				ARFFToMatlab.salvar(nameDir + "/iTrain" + strni + ".dat",ARFFToMatlab.carregar(nameDir + "/iTrain" + strni + ".arff"),false);
				ARFFToMatlab.salvar(nameDir + "/iTest" + strni + ".dat",ARFFToMatlab.carregar(nameDir + "/iTest" + strni + ".arff"),false);
				//if (strategyCluEns == 3){ ARFFToMatlab.salvar("clustering/iTest.dat",ARFFToMatlab.carregar("files_" + nameData2.substring(0, nameData2.indexOf('-')) + "/iTest" + strni + ".arff"),false); }
			}

			AnalysingMemory(printResults);

			// ----------------------------------------------------------------
			// RUNNING SUPERVISED MODELS (sum of the typeClaEns vector > 0)
			// ----------------------------------------------------------------
                                              
                        int runClassEns = 0;
			runClassEns = typeClaEns.get(0) + typeClaEns.get(1) + typeClaEns.get(2) + typeClaEns.get(3) + typeClaEns.get(4) + typeClaEns.get(5) + typeClaEns.get(6) + typeClaEns.get(7) + typeClaEns.get(8);
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

						CrossValidationFolds cvft = new CrossValidationFolds(iTrainSVM, numFoldsGSSVM, 0);
						iTrainSVM = cvft.getTestData(0); 
						iTestSVM = cvft.getTrainData(0);

						GridSearchThreadSVM svmGS = new GridSearchThreadSVM(iTrainSVM, iTestSVM, properties.get(3));
						svmGS.runGridSearch();
						double bestCost = svmGS.getBestCost();
						double bestGamma = svmGS.getBestGamma();

						if (logFile){
							StringBuffer buf = new StringBuffer();
							buf.append(bestCost).append("Cost=\t Gamma=").append(bestGamma).append("\t Acc=").append(svmGS.getBestFit()).append("\n");
							ARFFToMatlab.salvar(nameDir + "/SVM_Grid-Search.dat", buf, true);
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
                                if (typeClaEns.get(4) == 1){
                                    
                                    BayesNet bayes = new BayesNet();
                                    
                                    vcls[contCla] = bayes;

                                    if (printResults > 0) {System.out.print("\n-> Bayes Net");}
                                    contCla++;
                                }
                                if (typeClaEns.get(5) == 1){
                                    
                                    Logistic log = new Logistic();
                                    
                                    vcls[contCla] = log;

                                    if (printResults > 0) {System.out.print("\n-> Logistic");}
                                    contCla++;
                                }
                                if (typeClaEns.get(6) == 1){
                                    
                                    MultilayerPerceptron mp = new MultilayerPerceptron();
                                    
                                    vcls[contCla] = mp;

                                    if (printResults > 0) {System.out.print("\n-> Multilayer Perceptron");}
                                    contCla++;
                                }
                                if (typeClaEns.get(7) == 1){
                                    
                                    SimpleLogistic sl = new SimpleLogistic();
                                    
                                    vcls[contCla] = sl;

                                    if (printResults > 0) {System.out.print("\n-> Simple Logistic");}
                                    contCla++;
                                }
                                if (typeClaEns.get(8) == 1){
                                    
                                    RandomForest rf = new RandomForest();
                                 
                                    vcls[contCla] = rf;

                                    if (printResults > 0) {System.out.print("\n-> Random Forest");}
                                    contCla++;
                                }
				if (printResults > 0) {System.out.print("\n\n");}

				// ---------------------------------------------------
				// RUNNING CLASSIFIER ENSEMBLE 
				// ---------------------------------------------------
				ArrayList<double[]> probClass = new ArrayList<>();
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
			// RUNNING UNSUPERVISED MODELS (numPartitions > 0)
			// ----------------------------------------------------------------
			if (numPartitions > 0){

				if (printResults > 0) {System.out.print("-> Building the cluster ensemble (strategy " + strategyCluEns + ")\n");}   

				// ---------------------------------------------------
				// remove class attribute
				// ---------------------------------------------------
				Remove remove;
				remove = new Remove();
                                
                                // when multilabel classification: more columns to remove
                                switch(nameData2){
                                    case "emotions-test":
                                        int [] selected1 = {72, 73, 74, 75, 76, 77};
                                        remove.setAttributeIndicesArray(selected1);
                                        break;
                                    case "flags-test":  
                                        int [] selected2 = {19, 20, 21, 22, 23, 24, 25};
                                        remove.setAttributeIndicesArray(selected2);
                                        break;
                                    case "mediamill-test": 
                                        int [] selected3 = {120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220};
                                        remove.setAttributeIndicesArray(selected3);
                                        break;
                                    case "scene-test":   
                                        int [] selected4 = {294, 295, 296, 297, 298, 299};
                                        remove.setAttributeIndicesArray(selected4);
                                        break;
                                    case "yeast-test": 
                                        int [] selected5 = {103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116};
                                        remove.setAttributeIndicesArray(selected5);
                                        break;
                                    default:
                                        remove.setAttributeIndices(String.valueOf(iTest.numAttributes()));
                                }
                                
				remove.setInvertSelection(new Boolean(false).booleanValue());
				remove.setInputFormat(iTest);
				iTest = Filter.useFilter(iTest, remove);
                                
                                // TESTANDO!!!
                                //ArffSaver saverTest3 = new ArffSaver();
				//saverTest3.setInstances(iTest);
				//saverTest3.setFile(new File(nameDir + "/iTest_RemovedClass.arff"));
				//saverTest3.writeBatch();
                                
				ArrayList<Integer> kMin = new ArrayList<>();
				ArrayList<Integer> kMax = new ArrayList<>();
				ArrayList<int[]> subAtt = null;

				/* if (strategyCluEns == 1){

					// ---------------------------------------------------
					// optimization of parameters (k)
					// ---------------------------------------------------
					int bestK = 0;
					double bestQuality = Double.NEGATIVE_INFINITY;

					ArrayList<Integer> nClusters = new ArrayList<>();

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
                                    // setting up unsupervised model
                                    // ---------------------------------------------------
                                    switch (numPartitions) {
                                        case 1:
                                            kMin.add(bestK*viewSize);
                                            kMax.add(bestK*viewSize);
                                            break;
                                        case 2:
                                            kMin.add(bestK*viewSize);
                                            kMin.add(bestK*2*viewSize);
                                            kMax.add(bestK*viewSize);
                                            kMax.add(bestK*2*viewSize);
                                            break;
                                        case 3:
                                            kMin.add(bestK*viewSize);
                                            kMin.add(bestK*2*viewSize);
                                            kMin.add(bestK*3*viewSize);
                                            kMax.add(bestK*viewSize);
                                            kMax.add(bestK*2*viewSize);
                                            kMax.add(bestK*3*viewSize);
                                            break;
                                        default:
                                            break;
                                    }
				}*/

				if (strategyCluEns == 2){

					int varKMin = iTrain.numClasses()*2;  // initial k = number of classes * 2
					int varKMax = (int)Math.round(Math.pow(iTest.numInstances(),0.25)); // final k = sqrt(sqrt(number instances)) //(int)Math.round(Math.sqrt(iTest.numInstances()));

					if (varKMax < 2){
						varKMax = 2;
					}
					if (varKMin > varKMax){
						varKMin = varKMax;
					}

					for (int i = 1; i<=numPartitions; ++i){kMin.add(varKMin);}	
					for (int i = 1; i<=numPartitions; ++i){kMax.add(varKMax);}	

					//generating subsets of attributes for each partition (with reposition)
					boolean subSetAtt = true;
					double[] pArray = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1};
					double pAtt = pArray[viewSize-1]; //using variable viewSize to get size of the subset

					if ((subSetAtt) && (iTest.numAttributes()>2) && ((iTest.numAttributes()*pAtt)*numPartitions<=iTest.numAttributes())){
						int maxAtt = iTest.numAttributes();             // total number o attributes
						int maxSubAtt = (int) Math.round(maxAtt*pAtt);  // number of attributes in the attribute subsets
						int iAtt[] = new int[maxAtt];
						subAtt = new ArrayList<int[]>();

						for (int i = 0; i<maxAtt; ++i){iAtt[i] = i;}	
						for (int j = 1; j<=numPartitions; ++j){
							int iSubAtt[] = new int[maxSubAtt];
							for (int i = 0; i<maxSubAtt; ++i){
								boolean inserted = true;
								while (inserted){ // getting subset of attribute indexes for each partition
									int rnd1 = RandomAttInd(maxAtt); // selecting indexes randomly
									iSubAtt[i] = rnd1;
									inserted = false;
									if (iAtt[rnd1] == -1){ // avoiding same index 
										inserted = true;
									}
									iAtt[rnd1] = -1;
								}
							}
							subAtt.add(iSubAtt); // subset of attributes for each partition
						}
					}
				}

				// Using ELKI package to cluster objects from iTest
				if (strategyCluEns == 3){
                                    
                                        ArffSaver saverTest2 = new ArffSaver();
                                        saverTest2.setInstances(iTest);
                                        saverTest2.setFile(new File("clustering/iTest.arff"));
                                        saverTest2.writeBatch();
 
                                        ARFFToMatlab.salvar("clustering/iTest.dat",ARFFToMatlab.carregar("clustering/iTest.arff"),false);

					String nFile = "clustering/iTest.dat";

					File fileTest = new File(nFile);
					if (fileTest.exists()) {

						int[] kArray = {2,3,5,6,8};
						for (int i = 0; i < numPartitions; i++) {

							String k = String.valueOf(kArray[i]);

							String cluster = String.valueOf(i+1);

							String runELKI1 = "java -jar lib/elki-bundle-0.7.1.jar KDDCLIApplication " +
									"-dbc.in clustering/iTest.dat " +
									"-algorithm clustering.kmeans.KMedoidsPAM " +
									"-algorithm.distancefunction CosineDistanceFunction " +
									"-kmeans.k ";

							String runELKI2 = " -kmeans.initialization RandomlyChosenInitialMeans " +
									"-resulthandler ResultWriter -out clustering/";
                                                        
                                                        int em = 0;
                                                        if (em == 1){
                                                            runELKI1 = "java -jar lib/elki-bundle-0.7.1.jar KDDCLIApplication " +
                                                                    "-dbc.in clustering/iTest.dat " +
                                                                    "-algorithm clustering.em.EM " +
                                                                    "-em.model MultivariateGaussianModelFactory " +
                                                                    "-em.centers KMeansPlusPlusInitialMeans " +
                                                                    "-em.k ";
                                                             
                                                            runELKI2 = "-resulthandler ResultWriter -out clustering/";
                                                        }

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
                                        int runs = 10;
                                        ClustererEnsemble clue = new ClustererEnsemble(kMin, kMax, runs, iTest, subAtt);
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
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ArrayIndexOutOfBoundsException e) {
			e.printStackTrace();
                } catch (IllegalArgumentException e){
                        e.printStackTrace();
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

        // MOSTRAR PARA DOUGLAS
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
			ClassLoader loader = RunEnsembleSuppliedTestSet.class.getClassLoader();
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