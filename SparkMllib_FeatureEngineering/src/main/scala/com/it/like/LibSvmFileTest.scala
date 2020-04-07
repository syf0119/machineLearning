package com.it.like

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object LibSvmFileTest {
  def main(args: Array[String]): Unit = {
    val sc= {
      val sparkConf: SparkConf = new SparkConf().setMaster("local[4]")
        .setAppName("LibSvmFileTest")
      val sparkContext = new SparkContext(sparkConf)
      sparkContext.setLogLevel("warn")
      sparkContext

    }

    val path="F:\\idea\\machineLearning\\SparkMllib_FeatureEngineering\\src\\main\\resources\\1.text"
   val labelPointRdd: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc,path)
 labelPointRdd.foreach(x=>{
   println(x.features)

 })
 labelPointRdd.foreach(x=>{
   println(x.label)

 })
 labelPointRdd.foreach(x=>{
   println(x)

 })
  }

}
