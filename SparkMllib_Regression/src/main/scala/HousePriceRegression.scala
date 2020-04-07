import org.apache.spark.SparkContext
import org.apache.spark.ml.{Pipeline, PipelineModel, Transformer}
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.{DataFrame, SparkSession}

object HousePriceRegression {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder()
      .appName("testMlSummary2")
      .master("local[4]")
      .getOrCreate()
    val sc: SparkContext = spark.sparkContext
    sc.setLogLevel("warn")
    val path="F:\\idea\\machineLearning\\SparkMllib_Regression\\src\\main\\resources\\house.txt"
    val data: DataFrame = spark.read.format("csv")
      .option("header", true)
        .option("sep",";")
      .load(path)
    val selectData: DataFrame = data.select(data("square").cast("double")
      , data("type")
      , data("price").cast("double"))
    selectData.printSchema()
    selectData.show(false)
    val stringIndexer: StringIndexer = new StringIndexer().setInputCol("type")
      .setOutputCol("type_index")
    val vectorAssembler: VectorAssembler = new VectorAssembler()
      .setInputCols(Array("square","type_index"))
      .setOutputCol("features")


    val linearRegression: LinearRegression = new LinearRegression()
      .setElasticNetParam(0.3)
      .setSolver("normal")
      .setTol(1E-6)
      .setMaxIter(100)
      .setLabelCol("price")
      .setFeaturesCol("features")


    val pipeline=new Pipeline().setStages(Array(stringIndexer,vectorAssembler,linearRegression))

val model: PipelineModel = pipeline.fit(selectData)
val transformer: Transformer = model.stages(2)


  }
}
