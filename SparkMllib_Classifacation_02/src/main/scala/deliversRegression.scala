import org.apache.spark.SparkContext
import org.apache.spark.ml.linalg
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel, LinearRegressionTrainingSummary}
import org.apache.spark.sql.{DataFrame, SparkSession}

object deliversRegression {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder()
      .appName("testMlSummary2")
      .master("local[4]")
      .getOrCreate()
    val sc: SparkContext = spark.sparkContext
    sc.setLogLevel("warn")

    val data: DataFrame = spark.createDataFrame(
      Seq(
        (9.3, Vectors.dense(100, 4)),
        (4.8, Vectors.dense(50, 3)),
        (8.9, Vectors.dense(100, 4)),
        (6.5, Vectors.dense(100, 2)),
        (4.2, Vectors.dense(50, 2)),
        (6.2, Vectors.dense(80, 2)),
        (7.4, Vectors.dense(75, 3)),
        (6.0, Vectors.dense(65, 4)),
        (7.6, Vectors.dense(90, 3)),
        (6.1, Vectors.dense(90, 2))
      )
    ).toDF("label", "features")
    val linearRegression: LinearRegression = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setMaxIter(20)
      .setStandardization(true)
    val model: LinearRegressionModel = linearRegression.fit(data)
    val coefficients: linalg.Vector = model.coefficients
    val intercept: Double = model.intercept
    println(coefficients)
    println(intercept)


    val summary: LinearRegressionTrainingSummary = model.summary
    println(summary.r2)
    println(summary.meanAbsoluteError)
    println(summary.meanSquaredError)
    println(summary.rootMeanSquaredError)

  }

}
