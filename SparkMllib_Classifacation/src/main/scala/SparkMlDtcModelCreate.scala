import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

object SparkMlDtcModelCreate {

  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder()
      .appName("testMlSummary2")
      .master("local[4]")
      .getOrCreate()
    val sc: SparkContext = spark.sparkContext
    sc.setLogLevel("warn")
    val data: DataFrame = spark.read.format("libsvm")
      .load("F:\\idea\\machineLearning\\SparkMllib_Classifacation\\src\\main\\resources\\sample_libsvm_data.txt")
    data.printSchema()
    data.show(false)
    val Array(trainSet, testSet): Array[Dataset[Row]] = data.randomSplit(Array(0.8, 0.2), 123L)

    //    * 5-准备算法----业务+数据----分类问题
    //      .setFeaturesCol()   设置特征列---一定来源于原始的数据中
    //      .setLabelCol()      设置标签列----一定来源于原始的数据中
    //      .setImpurity()      设置不纯度
    //    * Supported: "entropy"在ID3中的算法 and "gini"是Cart树中的算法.
    //      * (default = gini)--------Cart树中的算法
    //      .setMaxDepth()      设置树的最大的深度，用于降低模型过拟合风险，预剪枝
    //      .setPredictionCol() 设置预测列 --prediction
    val classifierData: DecisionTreeClassifier = new DecisionTreeClassifier()
      .setImpurity("entropy")
      .setFeaturesCol("features")
      .setMaxDepth(5)
      .setPredictionCol("pre_col")
      .setLabelCol("label")
    val treeClassificationModel: DecisionTreeClassificationModel = classifierData.fit(trainSet)
    val trainRes: DataFrame = treeClassificationModel.transform(trainSet)
    val testRes: DataFrame = treeClassificationModel.transform(testSet)
    val evaluator: MulticlassClassificationEvaluator = new MulticlassClassificationEvaluator().setLabelCol("label")
      .setPredictionCol("pre_col")
      .setMetricName("accuracy")
    val trainAccuracy: Double = evaluator.evaluate(trainRes)
    val testAccuracy: Double = evaluator.evaluate(testRes)
    println("train:" + trainAccuracy)
    println("test:" + testAccuracy)
    println(treeClassificationModel.toDebugString)

  }

}
