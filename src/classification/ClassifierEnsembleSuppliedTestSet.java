package classification;

import all.GridSearchThreadSVM;
import all.CrossValidationFolds;
import all.ClassifierEnsembleThread;
import all.ARFFToMatlab;
import weka.classifiers.*;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.trees.J48;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
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
public class ClassifierEnsembleSuppliedTestSet {

	//*************************************************************************
	// Main
	//*************************************************************************
	public static void main (String args[])
	{   
		
                String path_results = "results/";  

		ArrayList<Integer> typeClaEns = new ArrayList<>();
		typeClaEns.add(0); // NB
		typeClaEns.add(0); // J48
		typeClaEns.add(1); // KNN
		typeClaEns.add(0); // SVM
                typeClaEns.add(0); // BayesNet
                typeClaEns.add(0); // Logistic
                typeClaEns.add(0); // Multilayer Perceptron
                typeClaEns.add(0); // SimpleLogistic
                typeClaEns.add(0); // RandomForest
		
		//  0: for testing (to build the test and train sets); 
		//  1: for validation (fold 1 of 2 from labeled objects - the dataset's name appears with "R");
		//  2: for validation (fold 2 of 2 from labeled objects - the dataset's name appears with "R");
		int validation = 0;

		int printResults = 1;
                
                String sFCAE = "";
                if (validation > 0){
                        sFCAE = "_" + "R" + typeClaEns.get(0) + typeClaEns.get(1) + typeClaEns.get(2) + typeClaEns.get(3) + typeClaEns.get(4) + typeClaEns.get(5) + typeClaEns.get(6) + typeClaEns.get(7) + typeClaEns.get(8);
                }else{
                        sFCAE = "_" + typeClaEns.get(0) + typeClaEns.get(1) + typeClaEns.get(2) + typeClaEns.get(3) + typeClaEns.get(4) + typeClaEns.get(5) + typeClaEns.get(6) + typeClaEns.get(7) + typeClaEns.get(8);
                }
                  
                String trainData = ClassifierEnsembleSuppliedTestSet.class.getProtectionDomain().getCodeSource().getLocation().getPath() + "hcr_hashlex_testset.arff"; 
		String testData = ClassifierEnsembleSuppliedTestSet.class.getProtectionDomain().getCodeSource().getLocation().getPath() + "hcr_hashlex_trainset.arff"; 

		RunEnsembleSTS(trainData, 
			       testData,
			       path_results, 
			       typeClaEns, 
			       validation,
			       printResults, 
                               sFCAE);
	}

	//******************************************************************************
	// Creating files labels.dat, piSet.dat, and SSet.dat - sentiment analysis
	//******************************************************************************
	public static ArrayList<String> RunEnsembleSTS(String trainData, String testData, String path_results, ArrayList<Integer> typeClaEns, int validation, int printResults, String sFCAE)
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
                            trainData = ClassifierEnsembleSuppliedTestSet.class.getProtectionDomain().getCodeSource().getLocation().getPath() + "leaves-train.arff";
                        }
                        if (testData.equals("")){
                            testData = ClassifierEnsembleSuppliedTestSet.class.getProtectionDomain().getCodeSource().getLocation().getPath() + "leaves-test.arff";
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
                        result.add("labels_" + nameData1.substring(0, 5) + sFCAE + ".dat");
                        result.add("piSet_" + nameData1.substring(0, 5) + sFCAE + ".dat");

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
				FileWriter wLabels = new FileWriter(new File(path_results + result.get(0)),false);
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
				writer = new FileWriter(new File(path_results + result.get(1)),false);
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

			
		} catch (FileNotFoundException e) {
                    
		} catch (IOException e) {
			
		} catch (ArrayIndexOutOfBoundsException | IllegalArgumentException e) {
			
                } catch (Exception e) {
			
		} finally{}

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
			ClassLoader loader = ClassifierEnsembleSuppliedTestSet.class.getClassLoader();
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
			
		} catch (IOException e) {
			
		}

		return properties;
	}
}