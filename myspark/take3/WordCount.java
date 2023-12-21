import java.util.Arrays;
import java.util.regex.Pattern;
import scala.Tuple2; // Added import statement

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;

public class WordCount {

    private static final Pattern SPACE = Pattern.compile(" ");

    public static void main(String[] args) {
        
        // Create SparkSession
        SparkSession spark = SparkSession.builder()
                .appName("WordCount")
                .master("local[*]")
                .getOrCreate();

        // Create JavaSparkContext
        JavaSparkContext sc = new JavaSparkContext(spark.sparkContext());

        // Load input data from HDFS
        String inputPath = "hdfs://tarun:9000/user/input.txt";
        JavaRDD<String> lines = sc.textFile(inputPath);

        // Split each line into words
        JavaRDD<String> words = lines.flatMap(line -> Arrays.asList(SPACE.split(line)).iterator());

        // Count the occurrence of each word
        JavaRDD<String> wordCounts = words.mapToPair(word -> new Tuple2<>(word, 1))
                .reduceByKey((a, b) -> a + b) // Changed to sum Integers
                .map(tuple -> tuple._1() + ": " + tuple._2());

        // Save the word count output to HDFS
        String outputPath = "hdfs://tarun:9000/Countout/";
        wordCounts.saveAsTextFile(outputPath);

        // Stop the SparkContext
        sc.stop();
    }
}

