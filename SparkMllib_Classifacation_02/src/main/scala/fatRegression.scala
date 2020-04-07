import org.apache.spark.SparkContext
import org.apache.spark.ml.linalg
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.sql.{DataFrame, SparkSession}

object fatRegression {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder()
      .appName("testMlSummary2")
      .master("local[4]")
      .getOrCreate()
    val sc: SparkContext = spark.sparkContext
    sc.setLogLevel("warn")

    val data: DataFrame = spark.createDataFrame(
      Seq(
        (9.5, Vectors.dense(23)),
        (17.8, Vectors.dense(27)),
        (21.2, Vectors.dense(39)),
        (25.9, Vectors.dense(41)),
        (27.5, Vectors.dense(45)),
        (26.3, Vectors.dense(49)),
        (28.2, Vectors.dense(50)),
        (29.6, Vectors.dense(53)),
        (30.2, Vectors.dense(54)),
        (31.4, Vectors.dense(56)),
        (30.8, Vectors.dense(57)),
        (33.5, Vectors.dense(58)),
        (35.2, Vectors.dense(60)),
        (34.6, Vectors.dense(61))
      )
    ).toDF("label", "features")
    val linearRegression: LinearRegression = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
    val model: LinearRegressionModel = linearRegression.fit(data)
    val coefficients: linalg.Vector = model.coefficients
    val intercept: Double = model.intercept
    println(coefficients)
    println(intercept)


  }

}
