package com.it.like

import org.apache.spark.ml.linalg
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}

object testMlSummary2 {
  def main(args: Array[String]): Unit = {
    val sparkSession: SparkSession = SparkSession.builder()
      .appName("testMlSummary2")
      .master("local[4]")
      .getOrCreate()
    sparkSession.sparkContext.setLogLevel("warn")
    import sparkSession.implicits._
    val vectors: Seq[linalg.Vector] = Seq(
      Vectors.sparse(4, Seq((0, 1.0), (3, -2.0))),
      Vectors.dense(4.0, 5.0, 0.0, 3.0),
      Vectors.dense(6.0, 7.0, 0.0, 8.0),
      Vectors.sparse(4, Seq((0, 9.0), (3, 1.0))))
  val tuples: Seq[Tuple1[linalg.Vector]] = vectors.map(Tuple1.apply)
    val dataFrame: DataFrame = tuples.toDF("features")
    dataFrame.printSchema()
    dataFrame.show()
    val result  : DataFrame = Correlation.corr(dataFrame,"features","pearson")
    result.show()

  }

}
