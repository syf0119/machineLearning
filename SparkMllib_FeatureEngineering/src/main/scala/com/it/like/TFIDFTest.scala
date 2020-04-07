package com.it.like

import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.{HashingTF, IDF, IDFModel, Tokenizer}
import org.apache.spark.sql.{DataFrame, SparkSession}

object TFIDFTest {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder()
      .appName("testMlSummary2")
      .master("local[4]")
      .getOrCreate()
    val sc: SparkContext = spark.sparkContext
    sc.setLogLevel("warn")


    val data: DataFrame = spark.createDataFrame(Seq((0, "Hi I heard about Spark"),
      (0, "I wish Java could use case classes"),
      (1, "Logistic regression models are neat"))).toDF("label", "words")
    //    * 3-解析数据
    data.printSchema()
    data.show(false)

    val tokenizer: Tokenizer = new Tokenizer().setInputCol("words").setOutputCol("token_words")
    val tokenResult : DataFrame = tokenizer.transform(data)
    tokenResult.printSchema()
    tokenResult.show(true)

    val hashingTF: HashingTF = new HashingTF().setInputCol("token_words").setOutputCol("hashTF_words")
    val tfResult: DataFrame = hashingTF.transform(tokenResult)

    tfResult.printSchema()
    tfResult.show(false)

    val idfResult: IDF = new IDF().setInputCol("hashTF_words").setOutputCol("idf_words")
    val idFModel: IDFModel = idfResult.fit(tfResult)
    val result: DataFrame = idFModel.transform(tfResult)
    result.printSchema()
    result.show(false)


  }

}
