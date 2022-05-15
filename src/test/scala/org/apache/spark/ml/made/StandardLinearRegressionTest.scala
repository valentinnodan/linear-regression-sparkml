package org.apache.spark.ml.made

import breeze.linalg.{DenseMatrix, DenseVector, sum}
import com.google.common.io.Files
import org.scalatest._
import flatspec._
import matchers._
import org.apache.spark.mllib.linalg.{Matrices, Vector, Vectors}
import org.apache.spark.sql.Row
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, typedLit, udf}

import scala.util.Random
//import org.apache.spark.ml.linalg.{Matrices, Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

class StandardLinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

  val delta = 0.0001
  lazy val data: Dataset[_] = StandardLinearRegressionTest._data
  lazy val dataF: Dataset[_] = StandardLinearRegressionTest._dataF
  lazy val coefs: DenseVector[Double] = DenseVector(StandardLinearRegressionTest._coefs.toArray)
  lazy val res: DenseVector[Double] = StandardLinearRegressionTest._res
  lazy val makeRes: UserDefinedFunction = StandardLinearRegressionTest._makeRes

  "Model" should "predict input data" in {
    val model: StandardLinearRegressionModel = new StandardLinearRegressionModel(
      coefs = coefs
    ).setOutputCol("features")

    validateModel(model.transform(data))
  }


  "Estimator" should "calculate coefs" in {
    val estimator = new StandardLinearRegression()
      .setInputCol("features")
      .setOutputCol("features")
      .setLabelCol("y")
    val ds = dataF.withColumn("y", makeRes(col("features")))
    val model = estimator.fit(ds)
    validateCoefs(model.coefs)
  }


  private def validateModel(data: DataFrame) = {
    val vector: Array[Row] = data.select("features").collect()
//    val vector: Array[Double] = data.collect().map(_.getAs[org.apache.spark.mllib.linalg.DenseVector](0))

    vector.length should be(data.collect().length)

    vector.toVector(0).getDouble(0) should be(res(0) +- delta)
    for (i <- 0 until 10) {
      vector.toVector(i).getDouble(0) should be(res(i) +- delta)
    }
  }
  private def validateCoefs(currCoefs: DenseVector[Double]) = {
    for (i <- 0 until coefs.length) {
      currCoefs(i) should be(coefs(i) +- delta)
    }
  }

//  "Estimator" should "work after re-read" in {
//
//    val pipeline = new Pipeline().setStages(Array(
//      new StandardLinearRegression()
//        .setInputCol("features")
//        .setOutputCol("features")
//    ))
//
//    val tmpFolder = Files.createTempDir()
//
//    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)
//
//    val reRead = Pipeline.load(tmpFolder.getAbsolutePath)
//
//    val model = reRead.fit(data).stages(0).asInstanceOf[StandardLinearRegressionModel]
//
//    model.coefs(0) should be(vectors.map(_(0)).sum / vectors.length +- delta)
//    model.coefs(1) should be(vectors.map(_(1)).sum / vectors.length +- delta)
//
//    validateModel(model, model.transform(data))
//  }
//
//  "Model" should "work after re-read" in {
//
//    val pipeline = new Pipeline().setStages(Array(
//      new StandardLinearRegression()
//        .setInputCol("features")
//        .setOutputCol("features")
//    ))
//
//    val model = pipeline.fit(data)
//
//    val tmpFolder = Files.createTempDir()
//
//    model.write.overwrite().save(tmpFolder.getAbsolutePath)
//
//    val reRead: PipelineModel = PipelineModel.load(tmpFolder.getAbsolutePath)
//
//    validateModel(model.stages(0).asInstanceOf[StandardLinearRegressionModel], reRead.transform(data))
//  }
}

object StandardLinearRegressionTest extends WithSpark {

  lazy val _vectors = Seq(
    Vectors.dense(13.5, 12),
    Vectors.dense(-1, 0)
  )

  lazy val _data: DataFrame = {
    import sqlc.implicits._
    val matrixRows = _dataset.rowIter.toSeq.map(_.toArray)
    val ds = spark.sparkContext.parallelize(matrixRows.map(x => Tuple1(Vectors.dense(x)))).toDF("features")
    ds
  }

  lazy val _dataF: DataFrame = {
    import sqlc.implicits._
    val matrixRows = _dataset.rowIter.toSeq.map(x => Tuple1(x))
    val ds = matrixRows.toDF("features")

    ds
  }

  lazy val _makeRes: UserDefinedFunction = udf { x: Vector =>
//    val curr = x.toArray
    val coefs = DenseVector(_coefs.toArray)

    sum(x.asBreeze.toDenseVector * coefs(0 until coefs.length))
  }
  lazy val _coefs: Vector = Vectors.dense(1.5, 0.3, -0.7)
  lazy val _dataset: Matrix = Matrices.dense(10, 3, DenseMatrix.rand[Double](10, 3).toArray)
  lazy val _res: DenseVector[Double] = DenseVector(_dataset.multiply(_coefs.toDense).toArray)
}
