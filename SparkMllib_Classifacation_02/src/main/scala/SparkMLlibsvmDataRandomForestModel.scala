import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

object SparkMLlibsvmDataRandomForestModel {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder()
      .appName("testMlSummary2")
      .master("local[4]")
      .getOrCreate()
    val sc: SparkContext = spark.sparkContext
    sc.setLogLevel("warn")
    val path="F:\\idea\\machineLearning\\SparkMllib_Classifacation_02\\src\\main\\resources\\sample_libsvm_data.txt"
    val data: DataFrame = spark.read.format("libsvm")
      .load(path)
    data.printSchema()
    data.show(false)
    //    * 4-准备算法
    //      .setNumTrees()   默认值20----树的个数
    //      .setFeaturesCol() 特征列---需要使用数据整合好的
    //      .setPredictionCol()  预测列---用户自己定义的
    //      .setSubsamplingRate()  下采样---range (0, 1]
    //      .setFeatureSubsetStrategy()  特征采样，默认是auto，sqrt是分类，onethird是分类问题
    //      .setLabelCol()  标签列
    //      .setImpurity()  不纯度的度量---entropy和gini
    //      .setMaxDepth()
    val Array(trainSet,testSet): Array[Dataset[Row]] = data.randomSplit(Array(0.8,0.2),123L)
    val randomForestClassifier: RandomForestClassifier = new RandomForestClassifier()
      .setNumTrees(20)
      .setFeaturesCol("features")
      .setPredictionCol("pre_col")
      .setSubsamplingRate(0.8)
      .setLabelCol("label")
      .setImpurity("gini")
      .setMaxDepth(5)
      .setFeatureSubsetStrategy("sqrt")
    val model: RandomForestClassificationModel = randomForestClassifier.fit(trainSet)
    val tranData  : DataFrame = model.transform(testSet)
    val evaluator: BinaryClassificationEvaluator = new BinaryClassificationEvaluator()
      .setMetricName("areaUnderROC")
      .setLabelCol("label")
      .setRawPredictionCol("pre_col")
    val accuracy: Double = evaluator.evaluate(tranData)
    println(accuracy)




  }

}
