import org.apache.spark.SparkConf
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * DESC: 
  * Complete data processing and modeling process steps:
  * 1-准备环境
  * 2-准备数据
  * 3-读取数据
  * 4-数据基本信息
  * 5-准备算法
  * 6-模型训练
  * 7-模型预测
  */
object kmeansDataMLCluster {
  def main(args: Array[String]): Unit = {
    //    * 1-准备环境
    //1-准备环境
    val conf: SparkConf = new SparkConf().setAppName("HousePriceRegression").setMaster("local[*]")
    val spark: SparkSession = SparkSession.builder().config(conf).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    //    * 2-准备数据
    val datapath = "D:\\BigData\\Workspace\\SparkMachineLearningTest\\SparkMllib_BigDataSH16\\src\\main\\resources\\sample_kmeans_data.txt"
    //    * 3-读取数据
    val data: DataFrame = spark.read.format("libsvm").load(datapath)
    //    * 4-数据基本信息
    data.printSchema()
    // label +features
    //    * 5-准备算法
    //      .setInitMode()  'random' and 'k-means||'
    //      .setK()    Default: 2.
    //      .setPredictionCol()  预测列
    //      .setFeaturesCol()    特征列
    //      .setSeed(123L)  当随机选择聚类中心的时候需要制定随机数的种子
    val kmeans: KMeans = new KMeans()
      .setInitMode("k-means||")
      .setK(2)
      .setPredictionCol("prces")
      .setFeaturesCol("features")
      .setMaxIter(100)
      .setTol(0.001)

    //    * 6-模型训练
    val kmeansModel: KMeansModel = kmeans.fit(data)
    kmeansModel.transform(data).show(false)
    //    * 7-模型预测
    println("WSSSE:", kmeansModel.computeCost(data))
    println("clustering:")
    //    kmeansModel.clusterCenters.foreach(println(_))
    //    (WSSSE:,0.11999999999994547)
    //    clustering:
    //    [0.1,0.1,0.1]
    //    [9.1,9.1,9.1]
  }
}
