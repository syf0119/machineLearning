import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

object LovaDataModel {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder()
      .appName("testMlSummary2")
      .master("local[4]")
      .getOrCreate()
    val sc: SparkContext = spark.sparkContext
    sc.setLogLevel("warn")
val data=spark.read.format("csv")
  .option("header",true)
  .load("F:\\idea\\machineLearning\\SparkMllib_Classifacation_02\\src\\main\\resources\\lovedata.csv")

    val selectData: DataFrame = data.select(
      data("is_date").cast("double"),
      data("age").cast("double"),
      data("handsome").cast("double"),
      data("income").cast("double"),
      data("is_gongwuyuan").cast("double")
    )

    val assembler: VectorAssembler = new VectorAssembler().setInputCols(Array("age", "handsome", "income", "is_gongwuyuan"))
      .setOutputCol("features")
    val assemblerData : DataFrame = assembler.transform(selectData).select("features","is_date")
      assemblerData.printSchema()
assemblerData.show(false)

    val Array(trainSet,testSet): Array[Dataset[Row]] = assemblerData.randomSplit(Array(0.8,0.2),123L)
    val decisionTreeClassifier: DecisionTreeClassifier = new DecisionTreeClassifier()
      .setMaxDepth(5)
      .setImpurity("entropy")
      .setLabelCol("is_date")
      .setFeaturesCol("features")
      .setPredictionCol("pre_col")
    val classificationModel: DecisionTreeClassificationModel = decisionTreeClassifier.fit(trainSet)
    val decisionTreeTestData: DataFrame = classificationModel.transform(testSet)
    decisionTreeTestData.printSchema()
    decisionTreeTestData.show(false)
    val evaluator: BinaryClassificationEvaluator = new BinaryClassificationEvaluator()
      .setMetricName("areaUnderROC")
      .setLabelCol("is_date")
      .setRawPredictionCol("pre_col")
    val evaluateTestSet : Double = evaluator.evaluate(decisionTreeTestData)
    println(evaluateTestSet)
    println(classificationModel.toDebugString)
    println(classificationModel.featureImportances)



  }

}
