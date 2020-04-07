package com.it.like



import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

object WordCount {
  def main(args: Array[String]): Unit = {
  val vector: linalg.Vector = Vectors.dense(2.0,3.0,4.5)
    val point = LabeledPoint(1,vector)
    println(point.label)
    println(point.features)

    val vector2: linalg.Vector = Vectors.sparse(4,Array(1,2,3),Array(3.4,2.3,3.4))
    val point2 = LabeledPoint.apply(1,vector2)
    println(point2.features)
    println(point2.label)


//   val vector1: linalg.Vector = Vectors.dense(1,2,3,4,5)
//    println(vector1)
//    val vector2: linalg.Vector = Vectors.sparse(4,Array(0,2,3),Array(9,2,7))
//    println(vector2)
//
//    println(vector1(0))
  }

}
