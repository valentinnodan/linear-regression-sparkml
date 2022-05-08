package org.apache.spark.ml.made

import breeze.linalg.{DenseMatrix, DenseVector}
import com.google.common.io.Files
import org.scalatest._
import flatspec._
import matchers._
import org.apache.spark.mllib.linalg.{Matrices, Vectors, Vector}
//import org.apache.spark.ml.linalg.{Matrices, Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

class StandardLinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

  val delta = 0.0001
  lazy val data: Dataset[_] = StandardLinearRegressionTest._data
  lazy val vectors: Seq[Vector] = StandardLinearRegressionTest._vectors
  lazy val coefs: DenseVector[Double] = DenseVector(StandardLinearRegressionTest._coefs)

  "Model" should "predict input data" in {
    val model: StandardLinearRegressionModel = new StandardLinearRegressionModel(
      coefs = coefs
    )

    validateModel(model.transform(data))
  }


  "Estimator" should "calculate coefs" in {
    val estimator = new StandardLinearRegression()
      .setInputCol("features")
      .setOutputCol("features")

    val model = estimator.fit(data)

    model.coefs(0) should be(vectors.map(_(0)).sum / vectors.length +- delta)
    model.coefs(1) should be(vectors.map(_(1)).sum / vectors.length +- delta)
  }


  private def validateModel(data: DataFrame) = {
    val vector: Array[Double] = data.collect().map(_.getAs[Double](0))

    vector.length should be(2)

    vector(0) should be(13.5 +- delta)
    vector(1) should be(12.0 +- delta)

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
    spark.sparkContext.parallelize(matrixRows).toDF()
  }

  lazy val _coefs: Vector = Vectors.dense(1.5, 0.3, -0.7)
  lazy val _dataset: Matrix = Matrices.dense(100000, 3, DenseMatrix.rand[Double](100000, 3).toArray)
  lazy val _res: DenseVector[Double] = DenseVector(_dataset.multiply(_coefs))
}
