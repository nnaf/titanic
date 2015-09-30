package titanic

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.types._
import org.apache.spark.sql.SQLImplicits
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

import org.apache.spark.ml.feature.{VectorAssembler, OneHotEncoder}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.{RandomForestClassifier, RandomForestClassificationModel}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}


object Titanic {
	def main(args: Array[String]) {
		val sc = new SparkContext(new SparkConf().setAppName("Titanic"))
		val sqlContext = new org.apache.spark.sql.SQLContext(sc)
		
		val df = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").load("src/main/resources/train.csv")
		val df2 = df.drop("Name").drop("Ticket").drop("PassengerId").drop("Cabin").filter( df("Embarked") !=="")

		val cols = df2.columns map {
			case "Fare" => df("Fare").cast(DoubleType).as("Fare")
			case "Survived" => df("Survived").cast(DoubleType).as("label")
			case "Age" => df("Age").cast(IntegerType).as("Age")
			case "Pclass" => df("Pclass").cast(DoubleType).as("Pclass")
			case "SibSp" => df("SibSp").cast(DoubleType).as("SibSp")
			case "Parch" => df("Parch").cast(DoubleType).as("Parch")
			case autre => df(autre)
		}

		val df3 = df2.select(cols :_*)

		val mean_age = df3.select(mean("Age")).first.getDouble(0)
		val df4 = df3.na.fill(Map("Age"-> mean_age))




		val Array(train, test) = df4.randomSplit(Array(0.75,0.25))

		val pclassEnc = new OneHotEncoder().setInputCol("Pclass").setOutputCol("Pclass_cat")
		val sibspEnc = new OneHotEncoder().setInputCol("SibSp").setOutputCol("SibSp_cat")
		val parchEnc = new OneHotEncoder().setInputCol("Parch").setOutputCol("Parch_cat")
		val sexEnc = new StringIndexer().setInputCol("Sex").setOutputCol("Sex_cat")
		val embarkedEnc = new StringIndexer().setInputCol("Embarked").setOutputCol("Embarked_cat")

		val vecAss = new VectorAssembler().setInputCols(Array("Pclass_cat","Sex_cat","Embarked_cat", "Age","Fare")).setOutputCol("features")

		val lr = new LogisticRegression()
		val lrp = new Pipeline().setStages(Array(pclassEnc,sibspEnc,parchEnc,sexEnc,embarkedEnc,vecAss, lr))
		val cvlr = new CrossValidator().setEstimator(lrp).setEvaluator(new BinaryClassificationEvaluator())

		val paramGrid = new ParamGridBuilder()
			.addGrid(lr.regParam, Array(1,0.1,0.01))
			.addGrid(lr.maxIter, Array(10,50,100))
			.build()
		cvlr.setEstimatorParamMaps(paramGrid)
		cvlr.setNumFolds(3)

		val cvlrmodel = cvlr.fit(train)

		for {s <- cvlrmodel.bestModel.asInstanceOf[PipelineModel].stages}
			println(s.explainParams())


		val ypred = cvlrmodel.transform(test)
		val perf = new BinaryClassificationEvaluator().setMetricName("areaUnderROC")
		println(s"${perf.getMetricName}: ${perf.evaluate(ypred)}")



		val labelEnc = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel")

		val rf = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("features").setNumTrees(10)
		val rfp = new Pipeline().setStages(Array(labelEnc,pclassEnc,sibspEnc,parchEnc,sexEnc,embarkedEnc,vecAss,rf))

		val rfmodel = rfp.fit(train)
		val ypred_rf = rfmodel.transform(test)
		val perf_rf = new BinaryClassificationEvaluator().setMetricName("areaUnderROC")
		println(s"${perf_rf.getMetricName}: ${perf_rf.evaluate(ypred_rf)}")


		val cvrf = new CrossValidator().setEstimator(rfp).setEvaluator(new BinaryClassificationEvaluator())

		val paramGridrf = new ParamGridBuilder()
			.addGrid(rf.maxDepth, Array(5,10,20))
			.addGrid(rf.numTrees, Array(50,100,500))
			.build()
		cvrf.setEstimatorParamMaps(paramGridrf)
		cvrf.setNumFolds(3)


		val cvrfmodel = cvrf.fit(train)

		for {s <- cvrfmodel.bestModel.asInstanceOf[PipelineModel].stages}
			println(s.explainParams())

		val ypredcvrf = cvrfmodel.transform(test)
		val perfcvrf = new BinaryClassificationEvaluator().setMetricName("areaUnderROC")
		println(s"${perfcvrf.getMetricName}: ${perfcvrf.evaluate(ypredcvrf)}")
	}
}






















