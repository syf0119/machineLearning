package com.it.like

import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.{Bucketizer, QuantileDiscretizer}
import org.apache.spark.sql.{DataFrame, SparkSession}



object BinarizerTest {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder()
      .appName("testMlSummary2")
      .master("local[4]")
      .getOrCreate()
    val sc: SparkContext = spark.sparkContext
    sc.setLogLevel("warn")
    val data = Array(-0.7,-0.5, -0.3, 0.0, 0.2,0.7)
    val source  : DataFrame = spark.createDataFrame(data.map(Tuple1.apply(_))).toDF("features")
    //val bucketizer: Bucketizer = new Bucketizer().setInputCol("features").setOutputCol("bucket_features").setSplits(Array(Double.NegativeInfinity,-0.5,0,0.5,Double.PositiveInfinity))

//    val result: DataFrame = bucketizer  .transform(source)
    val quantileDiscretizer: QuantileDiscretizer = new QuantileDiscretizer().setInputCol("features").setOutputCol("dis_res").setNumBuckets(3)
    val bucketizer: Bucketizer = quantileDiscretizer.fit(source)
val result: DataFrame = bucketizer.transform(source)
    source.printSchema()
    source.show()
    result.printSchema()
    result.show()
  }

}
