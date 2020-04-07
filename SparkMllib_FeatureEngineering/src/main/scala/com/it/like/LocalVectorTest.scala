package com.it.like

import org.apache.spark.mllib.linalg.{Matrices, Matrix}

object LocalVectorTest {
  def main(args: Array[String]): Unit = {
  val matrix: Matrix = Matrices.dense(3,2,Array(1,2,3,10,20,30))
  val sm: Matrix = Matrices.sparse(3,2, Array(1,2,3), Array(0, 2,1), Array(9, 6,8))
  println(matrix)
  println(sm)
}

}
