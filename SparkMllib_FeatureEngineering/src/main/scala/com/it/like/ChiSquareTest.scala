package com.it.like

import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.ChiSqSelector
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{DataFrame, SparkSession}

object ChiSquareTest {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder()
      .appName("testMlSummary2")
      .master("local[4]")
      .getOrCreate()
    val sc: SparkContext = spark.sparkContext
    sc.setLogLevel("warn")

    val data = Seq(
      (7, Vectors.dense(0.0, 0.0, 18.0, 1.0), 1.0),
      (8, Vectors.dense(0.0, 1.0, 12.0, 0.0), 0.0),
      (9, Vectors.dense(1.0, 0.0, 15.0, 0.1), 0.0)
    )
import spark.implicits._
    val dataSource: DataFrame = spark.createDataset(data).toDF("id", "features", "clicked")
    dataSource.printSchema()
    dataSource.show(false)

    val chiSqSelector: ChiSqSelector = new ChiSqSelector().setLabelCol("clicked").setFeaturesCol("features").setNumTopFeatures(2)
    val result  : DataFrame = chiSqSelector.fit(dataSource).transform(dataSource)
    result.printSchema()
    result.show(false)
  }

}
