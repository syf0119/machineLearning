import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.{DataFrame, SparkSession}

object LibSvmDataSetLRPipeline {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder()
      .appName("testMlSummary2")
      .master("local[4]")
      .getOrCreate()
    val sc: SparkContext = spark.sparkContext
sc.setLogLevel("WARN")


    val path="F:\\idea\\machineLearning\\SparkMllib_Regression\\src\\main\\resources\\sample_libsvm_data.txt"

    val data: DataFrame = spark.read.format("libsvm")
      .load(path)
    data.printSchema()
    data.show(false)

    val logisticRegression: LogisticRegression = new LogisticRegression()
      .setMaxIter(20)
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setElasticNetParam(0.5)
      .setThreshold(0.5)
      .setPredictionCol("pre_col")
    val tran_data : DataFrame = logisticRegression.fit(data).transform(data)
    val evaluator: MulticlassClassificationEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setMetricName("accuracy")
      .setPredictionCol("pre_col")
    val accuracy  : Double = evaluator.evaluate(tran_data)
    println(accuracy)
  }

}
