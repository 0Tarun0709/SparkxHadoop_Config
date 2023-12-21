import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.StructType;

public class JavaLogisticRegressionExample {

    public static void main(String[] args) {
        // Create a SparkSession
        SparkSession spark = SparkSession.builder()
                .appName("JavaLogisticRegressionExample")
                .getOrCreate();

        // Read the CSV file from HDFS
        StructType schema = new StructType()
                .add("sepal_length", "double")
                .add("sepal_width", "double")
                .add("petal_length", "double")
                .add("petal_width", "double")
                .add("class", "string");
        Dataset<Row> data = spark.read()
                .schema(schema)
                .format("csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .load("hdfs://tarun:9000/user/heart.csv");
                
                data.show();

        // Convert the "class" column to numerical labels
        StringIndexer indexer = new StringIndexer()
                .setInputCol("class")
                .setOutputCol("label");
        Dataset<Row> indexed = indexer.fit(data).transform(data);

        // Assemble the feature columns into a single vector column
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"sepal_length", "sepal_width", "petal_length", "petal_width"})
                .setOutputCol("features");
        Dataset<Row> featureData = assembler.transform(indexed).select("label", "features");

        // Split the data into training and test sets
        Dataset<Row>[] splits = featureData.randomSplit(new double[]{0.7, 0.3});
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];

        // Train a logistic regression model
        LogisticRegression lr = new LogisticRegression()
                .setLabelCol("label")
                .setFeaturesCol("features")
                .setMaxIter(10);
        LogisticRegressionModel model = lr.fit(trainingData);

        // Make predictions on the test set
        Dataset<Row> predictions = model.transform(testData);

        // Evaluate the model using binary classification metrics
        BinaryClassificationEvaluator evaluator = new BinaryClassificationEvaluator()
                .setLabelCol("label")
                .setRawPredictionCol("rawPrediction")
                .setMetricName("areaUnderROC");
        double areaUnderROC = evaluator.evaluate(predictions);

        System.out.println("Area under ROC = " + areaUnderROC);

        // Stop the SparkSession
        spark.stop();
    }
}


/*
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.StructType;

public class JavaLogisticRegressionExample {

    public static void main(String[] args) {
        // Create a SparkSession
        SparkSession spark = SparkSession.builder()
                .appName("JavaLogisticRegressionExample")
                .getOrCreate();

        // Read the CSV file from HDFS
        StructType schema = new StructType()
                .add("sepal_length", "double")
                .add("sepal_width", "double")
                .add("petal_length", "double")
                .add("petal_width", "double")
                .add("class", "string");
        Dataset<Row> data = spark.read()
                .schema(schema)
                .format("csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .load("hdfs://tarun:9000/user/IRIS.csv");
                
                data.show();

        // Convert the "class" column to numerical labels
        StringIndexer indexer = new StringIndexer()
                .setInputCol("class")
                .setOutputCol("label");
        Dataset<Row> indexed = indexer.fit(data).transform(data);

        // Assemble the feature columns into a single vector column
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"sepal_length", "sepal_width", "petal_length", "petal_width"})
                .setOutputCol("features");
        Dataset<Row> featureData = assembler.transform(indexed).select("label", "features");

        // Split the data into training and test sets
        Dataset<Row>[] splits = featureData.randomSplit(new double[]{0.7, 0.3});
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];

        // Train a logistic regression model
        /*LogisticRegression lr = new LogisticRegression()
                .setLabelCol("label")
                .setFeaturesCol("features")
                .setMaxIter(10)
                .setFamily("multinomial");
        LogisticRegression lr = new LogisticRegression()
        .setLabelCol("label")
        .setFeaturesCol("features")
        .setMaxIter(10)
        .setFamily("multinomial");
 // Change setNumClasses() to setFamily()
        LogisticRegressionModel model = lr.fit(trainingData);

        // Make predictions on the test set
        Dataset<Row> predictions = model.transform(testData);
	predictions.printSchema();
        // Evaluate the model using binary classification metrics
        BinaryClassificationEvaluator evaluator = new BinaryClassificationEvaluator()
                .setLabelCol("label")
                .setRawPredictionCol("rawPrediction")
                .setFamily("multinomial")
                .setMetricName("areaUnderROC");
                
                
                
                

        double areaUnderROC = evaluator.evaluate(predictions);

        System.out.println("Area under ROC = " + areaUnderROC);

        // Stop the SparkSession
        spark.stop();
    }
}


/*
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.IndexToString;

public class JavaLogisticRegressionExample {

    public static void main(String[] args) {

        // Create a Spark session
        SparkSession spark = SparkSession.builder()
                .appName("JavaLogisticRegressionExample")
                .master("local[*]")
                .getOrCreate();

        // Load the dataset
        Dataset<Row> data = spark.read().format("csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .load("hdfs://tarun:9000/user/IRIS.csv");

        // Convert the categorical variable 'class' to numerical labels
        StringIndexer indexer = new StringIndexer()
                .setInputCol("class")
                .setOutputCol("label")
                .fit(data);
        Dataset<Row> indexedData = indexer.transform(data);

        // Prepare the features and labels for training
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"sepal_length", "sepal_width", "petal_length", "petal_width"})
                .setOutputCol("features");
        Dataset<Row> trainingData = assembler.transform(indexedData)
                .select("features", "label");

        // Split the data into training and testing sets
        Dataset<Row>[] splits = trainingData.randomSplit(new double[] {0.8, 0.2}, 1234);
        Dataset<Row> training = splits[0];
        Dataset<Row> testing = splits[1];

        // Train the logistic regression model
        LogisticRegression lr = new LogisticRegression()
                .setMaxIter(100)
                .setRegParam(0.3)
                .setElasticNetParam(0.8)
                .setLabelCol("label")
                .setFeaturesCol("features");
        lr.fit(training);

        // Make predictions on the testing set
        Dataset<Row> predictions = lr.transform(testing);

        // Convert the numerical labels back to categorical variable 'class'
        IndexToString converter = new IndexToString()
                .setInputCol("prediction")
                .setOutputCol("predicted_class")
                .setLabels(indexer.labels());
        Dataset<Row> convertedPredictions = converter.transform(predictions);

        // Evaluate the model
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");
        double accuracy = evaluator.evaluate(convertedPredictions);
        System.out.println("Accuracy: " + accuracy);

        // Stop the Spark session
        spark.stop();
    }
}

*/


