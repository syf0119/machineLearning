package com.it.like

import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.sql.{DataFrame, SparkSession}

object StandScalerTest {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder()
      .appName("testMlSummary2")
      .master("local[4]")
      .getOrCreate()
    val sc: SparkContext = spark.sparkContext
    sc.setLogLevel("warn")
val path="F:\\idea\\machineLearning\\SparkMllib_FeatureEngineering\\src\\main\\resources\\sample_libsvm_data.txt"
    val dataFrame: DataFrame = spark.read.format("libsvm")
      .load(path)
    dataFrame.printSchema()
    dataFrame.show(false)
    val standardScaler: StandardScaler = new StandardScaler().setInputCol("features").setWithMean(true).setWithStd(true)
      .setOutputCol("feature_res")
    val result  : DataFrame = standardScaler.fit(dataFrame).transform(dataFrame)
    result.printSchema()
    result.show(false)



  }

}
