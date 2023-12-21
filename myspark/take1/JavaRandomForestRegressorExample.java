/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//package org.apache.spark.examples.ml;

// $example on$
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorIndexer;
import org.apache.spark.ml.feature.VectorIndexerModel;
import org.apache.spark.ml.regression.RandomForestRegressionModel;
import org.apache.spark.ml.regression.RandomForestRegressor;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.sql.types.DoubleType;
// $example off$

/*public class JavaRandomForestRegressorExample {
  public static void main(String[] args) {
    SparkSession spark = SparkSession
      .builder()
      .appName("JavaRandomForestRegressorExample")
      .getOrCreate();

    // $example on$
    // Load and parse the data file, converting it to a DataFrame.
    Dataset<Row> data = spark.read().format("csv")
            .option("header", "true")
            .option("inferSchema", "true")
            .load("hdfs://tarun:9000/user/hvac.csv");
            
            data.show();
            data.printSchema();


    // Automatically identify categorical features, and index them.
    // Set maxCategories so features with > 4 distinct values are treated as continuous.
    VectorIndexerModel featureIndexer = new VectorIndexer()
      .setInputCol("T_Supply")
      .setOutputCol("indexedEnergy")
      .setMaxCategories(4)
      .fit(data);
      data.printSchema();

    // Split the data into training and test sets (30% held out for testing)
    Dataset<Row>[] splits = data.randomSplit(new double[] {0.7, 0.3});
    Dataset<Row> trainingData = splits[0];
    Dataset<Row> testData = splits[1];

    // Train a RandomForest model.
    RandomForestRegressor rf = new RandomForestRegressor()
      .setLabelCol("Power")
      .setFeaturesCol("indexedEnergy");

    // Chain indexer and forest in a Pipeline
    Pipeline pipeline = new Pipeline()
      .setStages(new PipelineStage[] {featureIndexer, rf});

    // Train model. This also runs the indexer.
    PipelineModel model = pipeline.fit(trainingData);

    // Make predictions.
    Dataset<Row> predictions = model.transform(testData);

    // Select example rows to display.
    predictions.select("prediction", "Power", "T_Supply").show(5);

    // Select (prediction, true label) and compute test error
    RegressionEvaluator evaluator = new RegressionEvaluator()
      .setLabelCol("T_Supply")
      .setPredictionCol("prediction")
      .setMetricName("rmse");
    double rmse = evaluator.evaluate(predictions);
    System.out.println("Root Mean Squared Error (RMSE) on test data = " + rmse);

    RandomForestRegressionModel rfModel = (RandomForestRegressionModel)(model.stages()[1]);
    System.out.println("Learned regression forest model:\n" + rfModel.toDebugString());
    // $example off$

    spark.stop();
  }
}

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class JavaRandomForestRegressorExample {
    public static void main(String[] args) {
        // Create a Spark session
        SparkSession spark = SparkSession.builder()
                .appName("SparkJob")
                .master("local[*]")
                .getOrCreate();

        // Read the CSV file into a Dataset
        Dataset<Row> dataset = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv("hdfs://tarun:9000/user/hvac.csv");

        // Calculate the RMS and Efficiency
        double rms = dataset.selectExpr("sqrt(avg(power*power))").first().getDouble(0);
        double efficiency = dataset.selectExpr("sum(energy) / (count(*) * 1000)").first().getDouble(0);

        // Print the results
        System.out.println("RMS: " + rms);
        System.out.println("Efficiency: " + efficiency);
	System.out.println("Algo has successfully implemented");
        // Stop the Spark session
        spark.stop();
    }
}
*//*

// this code works

import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.Dataset;
import java.util.Collections;


public class  JavaRandomForestRegressorExample{

    public static void main(String[] args) {

        // Create SparkSession
        SparkSession spark = SparkSession.builder()
                .appName("LinearRegressionExample")
                .getOrCreate();

        // Read dataset from HDFS directory
        Dataset<Row> data = spark.read().format("csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .load("hdfs://tarun:9000/user/hvac.csv");

        // Create vector of input features
        String[] inputCols = {"T_Supply", "T_Return", "SP_Return", "T_Saturation", "T_Outdoor", "RH_Supply", "RH_Return", "RH_Outdoor"};
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(inputCols)
                .setOutputCol("features");

        Dataset<Row> inputData = assembler.transform(data)
                .select("features", "Power", "Energy");

        // Split data into training and test sets
        Dataset<Row>[] splits = inputData.randomSplit(new double[]{0.7, 0.3});
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];

        // Create Linear Regression Model
        LinearRegression lr = new LinearRegression()
                .setLabelCol("Power")
                .setFeaturesCol("features")
                .setMaxIter(10)
                .setRegParam(0.3)
                .setElasticNetParam(0.8);

        // Train the model using training data
        LinearRegressionModel lrModel = lr.fit(trainingData);

        // Predict the values of power and energy for year 2024
        double[] features = {55.0, 35.0, 40.0, 45.0, 20.0, 65.0, 55.0, 75.0};
        Vector denseVector = Vectors.dense(features);
        Row newRow = RowFactory.create(denseVector, null, null);
        Dataset<Row> newDataset = spark.createDataFrame(Collections.singletonList(newRow), inputData.schema());
        Dataset<Row> prediction = lrModel.transform(newDataset);

        System.out.println("Predicted power value for year 2024: " + prediction.select("prediction").head().getDouble(0));
        System.out.println("Predicted energy value for year 2024: " + 			prediction.select("prediction").head().getDouble(1));
	System.out.println("Algo has successfully implemented");
        // Stop the SparkSession
        spark.stop();
    }
}*/
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.Dataset;
import java.util.Collections;

public class JavaRandomForestRegressorExample {

    public static void main(String[] args) {

        // Create SparkSession
        SparkSession spark = SparkSession.builder()
                .appName("LinearRegressionExample")
                .getOrCreate();

        // Read dataset from HDFS directory
        Dataset<Row> data = spark.read().format("csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .load("hdfs://tarun:9000/user/hvac.csv");

        // Create vector of input features
        String[] inputCols = {"T_Supply", "T_Return", "SP_Return", "T_Saturation", "T_Outdoor", "RH_Supply", "RH_Return", "RH_Outdoor"};
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(inputCols)
                .setOutputCol("features");

        Dataset<Row> inputData = assembler.transform(data)
                .select("features", "Power", "Energy");

        // Split data into training and test sets
        Dataset<Row>[] splits = inputData.randomSplit(new double[]{0.7, 0.3});
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];

        // Create Linear Regression Model
        LinearRegression lr = new LinearRegression()
                .setLabelCol("Power")
                .setFeaturesCol("features")
                .setMaxIter(10)
                .setRegParam(0.3)
                .setElasticNetParam(0.8);

        // Train the model using training data
        LinearRegressionModel lrModel = lr.fit(trainingData);

        // Make predictions on test data
        Dataset<Row> predictions = lrModel.transform(testData);

        // Compute RMSE
        RegressionEvaluator evaluator = new RegressionEvaluator()
                .setLabelCol("Power")
                .setPredictionCol("prediction")
                .setMetricName("rmse");
        double rmse = evaluator.evaluate(predictions);

        System.out.println("Root Mean Squared Error (RMSE) = " + rmse);
        System.out.println("\n\n \t **Algo has successfully implemented** \n\n");

        // Stop the SparkSession
        spark.stop();
    }
}



