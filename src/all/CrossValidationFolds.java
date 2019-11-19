package all;

import weka.core.Instances;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

/**
 * Author: 	Luiz F. S. Coletta 
 * Email: 	luiz.fersc@gmail.com
 * Date: 	25/10/2011 
 * Update:	29/08/2012
 */
public class CrossValidationFolds {

	private Instances data = null;
	private int maxFolds = 0; 

	public CrossValidationFolds(String dataPath, int maxFolds, int seed)
	{
		BufferedReader reader = null;

		try
		{   
			this.maxFolds = maxFolds;
			
			reader = new BufferedReader(new FileReader(dataPath));
			data = new Instances(reader);
			data.setClassIndex(data.numAttributes()-1);
			reader.close();
			
			Random rand = new Random(seed); 
			Instances randData = new Instances(data); 
			randData.randomize(rand); 
			randData.stratify(maxFolds);
			data = randData;
		}
		catch (FileNotFoundException exFNF)
		{
			System.out.println("File not found!");
		}
		catch (IOException exIO)
		{
			System.out.println("IO exception");
		}
	}
	
	public CrossValidationFolds(Instances data, int maxFolds, int seed)
	{
		try
		{   
			this.maxFolds = maxFolds;
			
			data.setClassIndex(data.numAttributes()-1);
			
			Random rand = new Random(seed); 
			Instances randData = new Instances(data); 
			randData.randomize(rand); 
			randData.stratify(maxFolds);
			this.data = randData;
		}
		catch (Exception exExc)
		{
			System.out.println("Exception");
		}
	}

	public Instances getTrainData(int fold)
	{
		return data.trainCV(maxFolds, fold);
	}

	public Instances getTestData(int fold)
	{
		return data.testCV(maxFolds, fold);
	}
}
