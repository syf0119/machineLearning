import org.apache.spark.SparkContext
import org.apache.spark.sql.{DataFrame, SparkSession}

object irisKmeansClustring {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder()
      .appName("testMlSummary2")
      .master("local[4]")
      .getOrCreate()
    val sc: SparkContext = spark.sparkContext
    sc.setLogLevel("WARN")
val path="F:\\idea\\machineLearning\\SparkMllib_Regression\\src\\main\\resources\\iris.csv"
    val data: DataFrame = spark.read.format("csv")
        .option("inferschema",true)
      .option("header", true)
      .load(path)
    data.printSchema()
data.show(false)


  }

}
