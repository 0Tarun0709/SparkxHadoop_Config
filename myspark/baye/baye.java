/*import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.sql.DataSet;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.functions;
//import org.apache.spark.sql.*;


public class baye {

    public static void main(String[] args) {
        // Set up Spark configuration
        SparkConf conf = new SparkConf().setAppName("KMeansExample").setMaster("local[*]");
        JavaSparkContext sc = new JavaSparkContext(conf);
        SQLContext sqlContext = new SQLContext(sc);

        // Load the CSV file from HDFS
        DataFrame dataFrame = sqlContext.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv("hdfs://tarun:9000/user/IRIS.csv");

        // Parse the data into a format that can be used by the KMeans algorithm
        JavaRDD<Vector> parsedData = dataFrame
                .select("sepal_length", "sepal_width", "petal_length", "petal_width")
                .rdd()
                .map(row -> Vectors.dense(row.getDouble(0), row.getDouble(1), row.getDouble(2), row.getDouble(3)))
                .toJavaRDD();

        // Cluster the data into three classes using KMeans
        int numClusters = 3;
        int numIterations = 20;
        KMeansModel clusters = KMeans.train(parsedData.rdd(), numClusters, numIterations);

        // Print the cluster centers
        Vector[] centers = clusters.clusterCenters();
        System.out.println("Cluster Centers:");
        for (Vector center : centers) {
            System.out.println(center);
        }

        // Save the model to HDFS
        clusters.save(sc.sc(), "hdfs://tarun:9000/Count/kmeansOut/");
    }
}*/

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class KMeansExample {

    public static void main(String[] args) {

        // Create a SparkConf object and set the application name
        SparkConf conf = new SparkConf().setAppName("KMeansExample");

        // Create a JavaSparkContext object
        JavaSparkContext sc = new JavaSparkContext(conf);

        // Create a SparkSession object
        SparkSession spark = SparkSession.builder()
                .appName("KMeansExample")
                .config(conf)
                .getOrCreate();

        // Load the data from HDFS
        JavaRDD<String> data = sc.textFile(""hdfs://tarun:9000/user/IRIS.csv);
        data.show();

        // Convert the JavaRDD to a DataFrame
        Dataset<Row> df = spark.read().csv(data);

        // Create a vector of features
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"feature1", "feature2", "feature3", "feature4"})
                .setOutputCol("features");

        // Transform the DataFrame to include the feature vector
        Dataset<Row> transformedData = assembler.transform(df);

        // Create a KMeans object and set the parameters
        KMeans kmeans = new KMeans()
                .setK(3)
                .setSeed(1L);

        // Train the KMeans model
        KMeansModel model = kmeans.fit(transformedData);

        // Print the cluster centers
        System.out.println("Cluster Centers: ");
        for (Vector center : model.clusterCenters()) {
            System.out.println(center);
        }

        // Get the predictions
        Dataset<Row> predictions = model.transform(transformedData);

        // Display the predicted cluster for each data point
        System.out.println("Predictions: ");
        predictions.select("prediction").show();

        // Stop the SparkContext
        sc.stop();
    }
}

