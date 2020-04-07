package com.it.like

import org.apache.spark.mllib.random.RandomRDDs
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object RandomNumberTest {
  def main(args: Array[String]): Unit = {
    val sparkSession: SparkSession = SparkSession.builder()
      .appName("testMlSummary2")
      .master("local[4]")
      .getOrCreate()
    sparkSession.sparkContext.setLogLevel("warn")

    val normalRDD : RDD[Double] = RandomRDDs.normalRDD(sparkSession.sparkContext,10)
   // normalRDD.foreach(println(_))
    val result: RDD[Int] = sparkSession.sparkContext.parallelize(1 to 10)
    val rdd: RDD[Int] = result.sample(false,0.2,10)
    rdd.foreach(println(_))

  }

}
