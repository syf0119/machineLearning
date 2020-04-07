import org.apache.spark.SparkContext
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{ChiSqSelector, StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

object PipelineirisClaasifitionModelVersion {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder()
      .appName("testMlSummary2")
      .master("local[4]")
      .getOrCreate()
    val sc: SparkContext = spark.sparkContext
    sc.setLogLevel("warn")
    val data: DataFrame = spark.read.format("csv")
      .option("header", true)
      .load("F:\\idea\\machineLearning\\SparkMllib_Classifacation_02\\src\\main\\resources\\iris.csv")

    //sepal_length|sepal_width|petal_length|petal_width
    val selectData  : DataFrame = data.select(
      data("sepal_length").cast("double"),
      data("sepal_width").cast("double"),
      data("petal_length").cast("double"),
      data("petal_width").cast("double"),
      data("class")
    )

    val Array(trainSet,testSet): Array[Dataset[Row]] = selectData.randomSplit(Array(0.8,0.2),123L)


    val stringIndexerData: StringIndexer = new StringIndexer().setInputCol("class")
      .setOutputCol("label")


    val vectorAssembler: VectorAssembler = new VectorAssembler()
      .setInputCols(Array("sepal_length", "sepal_width", "petal_length", "petal_width"))
      .setOutputCol("features")

    val sqSelector: ChiSqSelector = new ChiSqSelector().setFeaturesCol("features")
        .setOutputCol("sq_col")
      .setNumTopFeatures(2)
      .setLabelCol("label")

    val standardScaler: StandardScaler = new StandardScaler().setInputCol("sq_col")
      .setOutputCol("std_features")

    val treeClassifier: DecisionTreeClassifier = new DecisionTreeClassifier().setImpurity("entropy")
      .setMaxDepth(5)
      .setLabelCol("label")
      .setFeaturesCol("std_features")
      .setPredictionCol("pre_col")


val pipeline: Pipeline = new Pipeline().setStages(Array(stringIndexerData,vectorAssembler,sqSelector,standardScaler,treeClassifier))
    val model: PipelineModel = pipeline.fit(trainSet)
    val testResult: DataFrame = model.transform(testSet)




    val evaluator=new MulticlassClassificationEvaluator().setPredictionCol("pre_col")
        .setLabelCol("label")
      .setMetricName("accuracy")
    val accuracy: Double = evaluator.evaluate(testResult)
    println(accuracy)
    val str: String = model.stages(4).explainParams()
    println(str)
    

  }

}
