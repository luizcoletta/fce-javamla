/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package data;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils;

/**
 *
 * @author luiz
 */
public class SamplingData {
    
    
    public void Subsampling(String dataPath, String nameData, int sizeTrainSet) throws Exception{
        
        boolean changeSets = true;
        //double txSizeValSet1 = 0.5;
        //double txSizeValSet2 = 0.5;
        int Tfold;
        boolean printResults = true;

        if (printResults && changeSets) {System.out.print("\n-> CREATING FILES (labels, piSet and SSet) - train and test sets were switched\n");}
        if (printResults && !changeSets) {System.out.print("\n-> CREATING FILES (labels, piSet and SSet)\n");}

        FileWriter writer = null;
        PrintWriter out = null;

        // ------------------------------------------------------------
        // BUILDING THE TRAIN AND TEST SETS
        // ------------------------------------------------------------

        // first one fold (same train and test sets)
        ConverterUtils.DataSource oneFold = new ConverterUtils.DataSource(dataPath);
        Instances iTrain = oneFold.getDataSet();
        Instances iTest; // = oneFold.getDataSet();

        // to obtain the correct number of folds for the required number of objects in the train set
        Tfold = (int) Math.ceil(iTrain.numInstances()/sizeTrainSet);

        GenTrainTestSets cvf = new GenTrainTestSets(dataPath, Tfold, 0);
        
        int foldNumber = 1;
        
        // DAQUI PRA FRENTE TEM QUE TER UM FOR PARA ITERAR 'foldNumber'

        // getting the train and test sets from a particular fold ('foldNumber') of {1, 2, ..., Tfold}
        if (changeSets){
            iTrain = cvf.getTestData(foldNumber-1); // switched to have less objects in train set than test set (iTrain = cvf.getTrainData(fold-1);)
            iTest = cvf.getTrainData(foldNumber-1); // switched to have more objects in test set than train set (iTest = cvf.getTestData(fold-1);)
        }else{
            iTrain = cvf.getTrainData(foldNumber-1);
            iTest = cvf.getTestData(foldNumber-1);
        }

        // if validation the train set is divided to create a validation set
        /* if (trainTest > 0){
            // to obtain the correct number of folds for the required number of objects in the train set
            int TfoldVS = -1;
            if (sizeTrainSet < 10){
                TfoldVS = (int) Math.ceil(iTrain.numInstances()/(Math.round(iTrain.numInstances()*txSizeValSet1)-1));
            }
            if (sizeTrainSet >= 10){
                TfoldVS = (int) Math.ceil(iTrain.numInstances()/(Math.round(iTrain.numInstances()*txSizeValSet2)-1));
            }
            GenTrainTestSets cvft = new GenTrainTestSets(iTrain, TfoldVS, 0);

            if (changeSets){
                iTrain = cvft.getTestData(trainTest-1); 
                iTest = cvft.getTrainData(trainTest-1);
            }else{
                iTrain = cvft.getTrainData(trainTest-1); 
                iTest = cvft.getTestData(trainTest-1);						
            }
        }*/

        iTrain.setClassIndex(iTrain.numAttributes()-1);
        iTest.setClassIndex(iTrain.numAttributes()-1);

        // ------------------------------------------------------------
        // CREATING FILES
        // ------------------------------------------------------------		

        (new File("files_" + nameData)).mkdirs();

        ArffSaver saverTrain = new ArffSaver();
        saverTrain.setInstances(iTrain);
        saverTrain.setFile(new File("files_" + nameData + "/train" + foldNumber + ".arff"));
        saverTrain.writeBatch();

        ArffSaver saverTest = new ArffSaver();
        saverTest.setInstances(iTest);
        saverTest.setFile(new File("files_" + nameData + "/test" + foldNumber + ".arff"));
        saverTest.writeBatch();

        Arff2Matrix.salvar("files_" + nameData + "/train" + foldNumber + ".dat", Arff2Matrix.carregar("files_" + nameData + "/train" + foldNumber + ".arff"),false);
        Arff2Matrix.salvar("files_" + nameData + "/test" + foldNumber + ".dat", Arff2Matrix.carregar("files_" + nameData + "/test" + foldNumber + ".arff"),false);
    
    
    }
    
}
