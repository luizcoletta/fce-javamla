package all;

import weka.clusterers.SimpleKMeans;
import weka.core.DistanceFunction;
import weka.core.Instances;
import weka.core.Utils;

public class SimplifiedSilhouette{

    public double quality(SimpleKMeans kmeans, Instances data, DistanceFunction df){

        double[] silh = new double[data.numInstances()];

        try{

            double a, b;

            double[] tmp = new double[kmeans.numberOfClusters()]; //hold the distance between the instance and the centroids
            int[] instanceAssignment = kmeans.getAssignments();   //cluster that each instance is assigned

            double[] clustersSize = kmeans.getClusterSizes();
            Instances centroids = kmeans.getClusterCentroids();

            for(int i=0; i<data.numInstances(); ++i){

                if(clustersSize[instanceAssignment[i]] == 1){  //singleton case
                    silh[i] = 0;
                    continue;
                }

                for(int k=0; k<kmeans.numberOfClusters(); ++k){

                    if(k == instanceAssignment[i]){ // to avoid counting the centroid itself in the extern distance 
                            tmp[k] = Double.POSITIVE_INFINITY;
                            continue;
                    }					

                    tmp[k] = df.distance(data.instance(i), centroids.instance(k));
                }

                a = df.distance(data.instance(i), centroids.instance(instanceAssignment[i])); //intern distance
                b = tmp[Utils.minIndex(tmp)]; //nearest neighbor distance (extern distance)

                silh[i] = (b-a)/(Math.max(a,b));				
            }

        }catch(Exception e){
            e.printStackTrace();
            System.out.println("Problem calculating simplified silhouette.\n" + e.getMessage());
        }

        return Utils.mean(silh);
    }
}