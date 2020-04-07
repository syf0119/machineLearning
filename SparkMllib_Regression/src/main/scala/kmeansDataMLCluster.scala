import org.apache.spark.SparkContext
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.sql.{DataFrame, SparkSession}

object kmeansDataMLCluster {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder()
      .appName("testMlSummary2")
      .master("local[4]")
      .getOrCreate()
    val sc: SparkContext = spark.sparkContext
    sc.setLogLevel("WARN")
val path="F:\\idea\\machineLearning\\SparkMllib_Regression\\src\\main\\resources\\sample_kmeans_data.txt"
    val data: DataFrame = spark.read.format("libsvm").load(path)
    data.printSchema()
    data.show(false)

    /**
      *  .setInitMode("k-means||")
      *.setK(2)
      *.setPredictionCol("prces")
      *.setFeaturesCol("features")
      *.setMaxIter(100)
      *.setTol(0.001)
      */
    val kMeans: KMeans = new KMeans().setFeaturesCol("features")
      .setInitMode("k-means||")
      .setK(2)
      .setPredictionCol("pre_col")
      .setTol(0.001)
      .setMaxIter(100)
    val model: KMeansModel = kMeans.fit(data)
    val result: DataFrame = model.transform(data)

    result.show(false)
    println(model.computeCost(data))


  }




}
