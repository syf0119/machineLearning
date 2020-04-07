import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * DESC: 
  * Complete data processing and modeling process steps:
  * 1-准备环境
  * 2-准备数据
  * 3-数据解析
  * 4-特征工程
  * 5-准备KMenas算法
  * 6-训练模型
  * 7-模型预测
  */
object kmeanDataCluster {
  def main(args: Array[String]): Unit = {
    //    * 1-准备环境
    val spark: SparkConf = new SparkConf().setAppName("kmeanDataCluster").setMaster("local[*]")
    val sc = new SparkContext(spark)
    sc.setLogLevel("WARN")
    val datapath = "D:\\BigData\\Workspace\\SparkMachineLearningTest\\SparkMllib_BigDataSH16\\src\\main\\resources\\kmeans_data.txt"
    //    * 2-准备数据
    val data: RDD[String] = sc.textFile(datapath)
    //    * 3-数据解析
    val featuresData: RDD[Vector] = data.map(x=>Vectors.dense(x.split(" ").map(_.toDouble)))
    //    * 4-特征工程
    //    * 5-准备KMenas算法
    val kmeansModel: KMeansModel = KMeans.train(featuresData,2,20,"k-means||")
    //    * 6-训练模型
    val wssse: Double = kmeansModel.computeCost(featuresData)
    //    * 7-模型预测
    println("Wssse value is:",wssse)
    kmeansModel.clusterCenters.foreach(println(_))
  }
}
