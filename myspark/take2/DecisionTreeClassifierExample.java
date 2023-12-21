import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class DecisionTreeClassifierExample {

    public static void main(String[] args) {

        // Create SparkSession
        SparkSession spark = SparkSession.builder()
                .appName("IrisDecisionTreeClassifier")
                .master("local[*]")
                .getOrCreate();

        // Load data from HDFS directory
        String dataPath = "hdfs://tarun:9000/user/IRIS.csv";
        Dataset<Row> data = spark.read().option("header", "true").option("inferSchema", "true").csv(dataPath);

        // Split the data into training and testing sets (70% training, 30% testing)
        Dataset<Row>[] splits = data.randomSplit(new double[] { 0.7, 0.3 });
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];

        // Prepare the data for training by assembling feature vectors and indexing the target column
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[] { "sepal_length", "sepal_width", "petal_length", "petal_width" })
                .setOutputCol("features");
        Dataset<Row> assembledTrainingData = assembler.transform(trainingData);
        Dataset<Row> assembledTestData = assembler.transform(testData);

        StringIndexerModel labelIndexer = new StringIndexer()
                .setInputCol("class")
                .setOutputCol("indexedLabel")
                .fit(assembledTrainingData);

        // Train a DecisionTreeClassifier model
        DecisionTreeClassifier dt = new DecisionTreeClassifier()
                .setLabelCol("indexedLabel")
                .setFeaturesCol("features");

        // Convert indexed labels back to original labels for evaluation
        IndexToString labelConverter = new IndexToString()
                .setInputCol("prediction")
                .setOutputCol("predictedLabel")
                .setLabels(labelIndexer.labels());

        // Chain indexers and model in a Pipeline
        Pipeline pipeline = new Pipeline()
                .setStages(new org.apache.spark.ml.PipelineStage[] { labelIndexer, dt, labelConverter });

        // Fit the pipeline to the training data
        PipelineModel model = pipeline.fit(assembledTrainingData);

        // Make predictions on the testing data
        Dataset<Row> predictions = model.transform(assembledTestData);

        // Evaluate the model's performance using a multi-class classification evaluator
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("indexedLabel")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");
        double accuracy = evaluator.evaluate(predictions);
        System.out.println("Accuracy = " + accuracy);

        // Save the model to HDFS
        /**String modelPath = "hdfs://<your-hdfs-host>:<port>/path/to/model";
        model.write().overwrite().save(modelPath);
*/
        spark.stop();
    }
}

