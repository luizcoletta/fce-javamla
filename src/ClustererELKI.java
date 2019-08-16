/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author luiz
 */
import de.lmu.ifi.dbs.elki.algorithm.clustering.kmeans.KMeansLloyd;
import de.lmu.ifi.dbs.elki.algorithm.clustering.kmeans.initialization.RandomUniformGeneratedInitialMeans;
import de.lmu.ifi.dbs.elki.data.Cluster;
import de.lmu.ifi.dbs.elki.data.Clustering;
import de.lmu.ifi.dbs.elki.data.NumberVector;
import de.lmu.ifi.dbs.elki.data.model.KMeansModel;
import de.lmu.ifi.dbs.elki.data.type.TypeUtil;
import de.lmu.ifi.dbs.elki.database.Database;
import de.lmu.ifi.dbs.elki.database.StaticArrayDatabase;
import de.lmu.ifi.dbs.elki.database.ids.DBIDIter;
import de.lmu.ifi.dbs.elki.database.ids.DBIDRange;
import de.lmu.ifi.dbs.elki.database.relation.Relation;
import de.lmu.ifi.dbs.elki.datasource.ArrayAdapterDatabaseConnection;
import de.lmu.ifi.dbs.elki.datasource.DatabaseConnection;
import de.lmu.ifi.dbs.elki.distance.distancefunction.minkowski.SquaredEuclideanDistanceFunction;
import de.lmu.ifi.dbs.elki.logging.LoggingConfiguration;
import de.lmu.ifi.dbs.elki.utilities.random.RandomFactory;

import de.lmu.ifi.dbs.elki.algorithm.clustering.kmeans.BestOfMultipleKMeans;
import de.lmu.ifi.dbs.elki.algorithm.clustering.kmeans.quality.KMeansQualityMeasure;
import de.lmu.ifi.dbs.elki.algorithm.clustering.kmeans.quality.WithinClusterMeanDistanceQualityMeasure;
import de.lmu.ifi.dbs.elki.datasource.MultipleObjectsBundleDatabaseConnection;
import de.lmu.ifi.dbs.elki.datasource.bundle.MultipleObjectsBundle;
import de.lmu.ifi.dbs.elki.datasource.parser.ArffParser;
import de.lmu.ifi.dbs.elki.evaluation.clustering.internal.EvaluateSimplifiedSilhouette;
import de.lmu.ifi.dbs.elki.evaluation.clustering.internal.NoiseHandling;
import java.io.FileInputStream;
import java.io.InputStream;
import java.util.Arrays;
import java.util.regex.Pattern;
//import de.lmu.ifi.dbs.elki.datasource.parser;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * Example program to generate a random data set, and run k-means on it.
 * 
 * @author Erich Schubert
 * @since 0.7.0
 */
public class ClustererELKI {
  /**
   * Main method
   * 
   * @param args Command line parameters (not supported)
   */
  public static void main(String[] args) throws Exception {
    // Set the logging level to statistics:
    LoggingConfiguration.setStatistics();
    
    String trainData = RunEnsembleSuppliedTestSet.class.getProtectionDomain().getCodeSource().getLocation().getPath() + "mastite.arff"; 

    // https://elki-project.github.io/releases/current/doc/de/lmu/ifi/dbs/elki/datasource/parser/NumberVectorLabelParser.html
    // NumberVectorLabelParser
    
    //DataSource oneFoldTrain = new DataSource(trainData);
    //Instances iTest = oneFoldTrain.getDataSet();
    
    InputStream dataARFF = new FileInputStream(trainData);
    
    ArffParser ap = new ArffParser("", "class");
    
    MultipleObjectsBundle d = ap.parse(dataARFF);
    
    
   // NumberVectorLabelParser nvp = new NumberVectorLabelParser(Pattern.quote("\\s*[,;\\s]\\s*"), Pattern.quote("\"'") , Pattern.quote("^\s*(#|//|;).*$"), )
    
    
    // Generate a random data set.
    // Note: ELKI has a nice data generator class, use that instead.
    //double[][] data = new double[1000][2];
    //for(int i = 0; i < data.length; i++) {
    //  for(int j = 0; j < data[i].length; j++) {
    //    data[i][j] = Math.random();
    //  }
    //}

    // Adapter to load data from an existing array.
    
   // DatabaseConnection dbc = new ArrayAdapterDatabaseConnection(d.getColumn(0));
    
    DatabaseConnection dbc = new MultipleObjectsBundleDatabaseConnection(d);
    //dbc.loadData(d); // = new ArrayAdapterDatabaseConnection(data);
    // Create a database (which may contain multiple relations!)
    Database db = new StaticArrayDatabase(dbc, null);
    // Load the data into the database (do NOT forget to initialize...)
    db.initialize();
    // Relation containing the number vectors:
    Relation<NumberVector> rel = db.getRelation(TypeUtil.NUMBER_VECTOR_FIELD);
    // We know that the ids must be a continuous range:
    DBIDRange ids = (DBIDRange) rel.getDBIDs();

    // K-means should be used with squared Euclidean (least squares):
    SquaredEuclideanDistanceFunction dist = SquaredEuclideanDistanceFunction.STATIC;
    // Default initialization, using global random:
    // To fix the random seed, use: new RandomFactory(seed);
    RandomUniformGeneratedInitialMeans init = new RandomUniformGeneratedInitialMeans(RandomFactory.DEFAULT);
    
    
    KMeansLloyd<NumberVector> km = new KMeansLloyd<>(dist, //
    3 /* k - number of partitions */, //
    0 /* maximum number of iterations: no limit */, init);
    
    KMeansQualityMeasure qm = new WithinClusterMeanDistanceQualityMeasure() ;
            
    BestOfMultipleKMeans<NumberVector,KMeansModel> bmr = new BestOfMultipleKMeans<>(5000, km, qm);
    
    
    
    Clustering<KMeansModel> c = bmr.run(db);
    
    EvaluateSimplifiedSilhouette ss = new EvaluateSimplifiedSilhouette(dist, NoiseHandling.TREAT_NOISE_AS_SINGLETONS, false);
    
    double result = ss.evaluateClustering(db, rel, c);
    
    System.out.println("Valor da Silhueta : " + String.format("%.2f", result));

            
    
    // Textbook k-means clustering:
    

    // K-means will automatically choose a numerical relation from the data set:
    // But we could make it explicit (if there were more than one numeric
    // relation!): km.run(db, rel);
    //Clustering<KMeansModel> c = km.run(db);

    // Output all clusters:
    int i = 0;
    for(Cluster<KMeansModel> clu : c.getAllClusters()) {
      // K-means will name all clusters "Cluster" in lack of noise support:
      System.out.println("#" + i + ": " + clu.getNameAutomatic());
      System.out.println("Size: " + clu.size());
      System.out.println("Center: " + Arrays.toString(clu.getModel().getPrototype()));
      // Iterate over objects:
      System.out.print("Objects: ");
      for(DBIDIter it = clu.getIDs().iter(); it.valid(); it.advance()) {
        // To get the vector use:
        // NumberVector v = rel.get(it);

        // Offset within our DBID range: "line number"
        final int offset = ids.getOffset(it);
        System.out.print(" " + offset);
        // Do NOT rely on using "internalGetIndex()" directly!
      }
      System.out.println();
      ++i;
    }
  }
}
