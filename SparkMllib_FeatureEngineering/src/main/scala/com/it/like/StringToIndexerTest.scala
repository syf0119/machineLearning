package com.it.like

import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.{IndexToString, OneHotEncoder, StringIndexer, StringIndexerModel}
import org.apache.spark.sql.{DataFrame, SparkSession}

object StringToIndexerTest {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder()
      .appName("testMlSummary2")
      .master("local[4]")
      .getOrCreate()
    val sc: SparkContext = spark.sparkContext
    sc.setLogLevel("warn")


    val dataFrame: DataFrame = spark.createDataFrame(Seq((0, "a"), (1, "b"), (2, "b"), (3, "a"), (4, "a"), (5, "c")))
      .toDF("id", "category")
    dataFrame.printSchema()
    dataFrame.show()
    val stringIndexer: StringIndexer = new StringIndexer().setInputCol("category").setOutputCol("index")
    val stringIndexerModel: StringIndexerModel = stringIndexer.fit(dataFrame)
    val result  : DataFrame = stringIndexerModel.transform(dataFrame)
    result.printSchema()
    result.show()
//    val selectData  : DataFrame = result.select("index")
//
//    val indexToString: IndexToString = new IndexToString().setInputCol("index").setOutputCol("before")
//    val result2 : DataFrame = indexToString.transform(selectData)
//    result2.printSchema()
//    result2.show(false)

    val oneHotEncoder: OneHotEncoder = new OneHotEncoder().setInputCol("index").setOutputCol("oneHot").setDropLast(false)
    val result1: DataFrame = oneHotEncoder.transform(result)
    result1.printSchema()
    result1.show()
  }
}
