package com.it.like

import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{DataFrame, SparkSession}

object FeaturesVectorAssemble {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder()
      .appName("testMlSummary2")
      .master("local[4]")
      .getOrCreate()
    val sc: SparkContext = spark.sparkContext
    sc.setLogLevel("warn")

    val dataFrame: DataFrame = spark.createDataFrame(
      Seq((0, 18, 1.0, Vectors.dense(0.0, 10.0, 0.5), 1.0),
        (1, 20, 2.0, Vectors.dense(0.1, 11.0, 0.5), 0.0))
    ).toDF("id", "hour", "mobile", "userFeatures", "clicked")
    dataFrame.printSchema()
    dataFrame.show()
      val vectorAssembler: VectorAssembler = new VectorAssembler().setInputCols(Array("hour","mobile","userFeatures")).setOutputCol("assemble_res")
    val result  : DataFrame = vectorAssembler.transform(dataFrame)
    result.printSchema()
    result.show(false)

  }
}