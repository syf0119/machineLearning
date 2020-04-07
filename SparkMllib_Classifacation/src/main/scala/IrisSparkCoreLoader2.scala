
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature._
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.sql.types.{DoubleType, StringType, StructType}
import org.apache.spark.sql.{DataFrame, SparkSession}

object IrisSparkCoreLoader2 {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder()
      .appName("testMlSummary2")
      .master("local[4]")
      .getOrCreate()
    val sc: SparkContext = spark.sparkContext
    sc.setLogLevel("warn")

    val dataFrame: DataFrame = spark.read.format("csv")
    .schema(new StructType().add("sepal_length", DoubleType, true)
      .add("sepal_width", DoubleType, true) //"double"
      .add("petal_length", DoubleType, true)
      .add("petal_width", DoubleType, true)
      .add("class_label", StringType, true))
        .load("F:\\idea\\machineLearning\\SparkMllib_Classifacation\\src\\main\\resources\\iris.data")
val indexerData: StringIndexer = new StringIndexer().setInputCol("class_label").setOutputCol("index_class_label")
   val tranData: DataFrame = indexerData.fit(dataFrame).transform(dataFrame)
//    tranData.printSchema()
//    tranData.show(100,false)
    val assemblerData: VectorAssembler = new VectorAssembler()
      .setInputCols(Array("sepal_length", "sepal_width", "petal_length", "petal_width"))
      .setOutputCol("features")
    val tranAssemblerData: DataFrame = assemblerData.transform(tranData)
    tranAssemblerData.printSchema()
    tranAssemblerData.show(false)
//    val resultMatrix  : DataFrame = Correlation.corr(tranAssemblerData,"features","pearson")
//    println("pearson:")
//resultMatrix.show(false)
//    val selectorData: ChiSqSelector = new ChiSqSelector().setFeaturesCol("features")
//    .setOutputCol("select_features")
//  .setLabelCol("index_class_label")
//      .setNumTopFeatures(3)
//    val selectorModel: ChiSqSelectorModel = selectorData.fit(tranAssemblerData)
//    val selectorTranData  : DataFrame = selectorModel.transform(tranAssemblerData)
//    selectorTranData.printSchema()
//    selectorTranData.show(false)

    val pca: PCA = new PCA().setInputCol("features").setOutputCol("pca_features").setK(2)
    val pcaTranData : DataFrame = pca.fit(tranAssemblerData).transform(tranAssemblerData)
    pcaTranData.printSchema()
    pcaTranData.show(false)

  }

}
