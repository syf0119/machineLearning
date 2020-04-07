package com.it.like
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.{Matrix, Vectors}
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object TestMlSummary {
  def main(args: Array[String]): Unit = {
    val sparkSession: SparkSession = SparkSession.builder().master("local[4]")
      .appName("TestMlSummary")
      .getOrCreate()
    val path="F:\\idea\\machineLearning\\SparkMllib_FeatureEngineering\\src\\main\\resources\\testSummary.txt"
    val source: RDD[String] = sparkSession.sparkContext.textFile(path)
    val vectorRdd = source.map(line => {
      val array = Array(line.split(",")(0).toDouble,line.split(",")(1).toDouble)
      Vectors.dense(array)
    })

    val data1: RDD[linalg.Vector] = sparkSession.sparkContext.parallelize(Seq(
      Vectors.dense(1.0, 10.0, 100.0,1000.0),
      Vectors.dense(2.0, 22.0, 204.0),
      Vectors.dense(3.0, 30.0, 300.0)
    ))

    val summary: MultivariateStatisticalSummary = Statistics.colStats( data1)

    println("min value:", summary.min)
    println("max value:", summary.max)
    println("mean value:", summary.mean)
    print("varience value:", summary.variance)
  }

}
