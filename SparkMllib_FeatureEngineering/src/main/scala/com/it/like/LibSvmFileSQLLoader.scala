package com.it.like

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}

object LibSvmFileSQLLoader {
  def main(args: Array[String]): Unit = {
    val sparkConf: SparkConf = new SparkConf().setAppName("LibSvmFileSQLLoader")
      .setMaster("local[4]")
    sparkConf
    val sparkSession: SparkSession = SparkSession.builder().config(conf = sparkConf).getOrCreate()
    sparkSession.sparkContext.setLogLevel("warn")
    import  sparkSession.implicits._

     val dataFrame: DataFrame = sparkSession.read.format("libsvm")
      .load("F:\\idea\\machineLearning\\SparkMllib_FeatureEngineering\\src\\main\\resources\\1.text")
    dataFrame.select($"features")
    .show(false)
  }

}
