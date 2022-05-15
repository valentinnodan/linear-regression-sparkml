package org.apache.spark.ml.made

import breeze.linalg.{DenseMatrix, DenseVector, sum}
import com.google.common.io.Files
import org.scalatest._
import flatspec._
import matchers._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.linalg.{Matrices, Vector, Vectors}
import org.apache.spark.sql.Row
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, typedLit, udf}
import org.apache.spark.ml.linalg.Matrix
import org.apache.spark.sql.{DataFrame, Dataset}

class StandardLinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

  val delta = 0.0001
  lazy val data: Dataset[_] = StandardLinearRegressionTest._data
  lazy val coefs: DenseVector[Double] = DenseVector(StandardLinearRegressionTest._coefs.toArray)
  lazy val res: DenseVector[Double] = StandardLinearRegressionTest._res
  lazy val makeRes: UserDefinedFunction = StandardLinearRegressionTest._makeRes

  "Model" should "predict input data" in {
    val model: StandardLinearRegressionModel = new StandardLinearRegressionModel(
      coefs = coefs
    ).setOutputCol("features")

    validateModel(model.transform(data))
  }


  "Estimator" should "calculate true coefs" in {
    import sqlc.implicits._
    val estimator = new StandardLinearRegression()
      .setInputCol("features")
      .setOutputCol("features")
      .setLabelCol("y")
      .setIter(10000)

    val ds = data.withColumn("y", makeRes(col("features")))
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

  "Estimator" should "work after re-read" in {

    val pipeline = new Pipeline().setStages(Array(
      new StandardLinearRegression()
        .setInputCol("features")
        .setOutputCol("features")
        .setLabelCol("y")
    ))
    val ds = data.withColumn("y", makeRes(col("features")))

    val tmpFolder = Files.createTempDir()

    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead = Pipeline.load(tmpFolder.getAbsolutePath)

    val model = reRead.fit(ds).stages(0).asInstanceOf[StandardLinearRegressionModel]

    validateCoefs(model.coefs)

    validateModel(model.transform(ds))
  }

  "Model" should "work after re-read" in {

    val pipeline = new Pipeline().setStages(Array(
      new StandardLinearRegression()
        .setInputCol("features")
        .setOutputCol("features")
        .setLabelCol("y")
    ))
    val ds = data.withColumn("y", makeRes(col("features")))

    val model = pipeline.fit(ds)

    val tmpFolder = Files.createTempDir()

    model.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead: PipelineModel = PipelineModel.load(tmpFolder.getAbsolutePath)

    validateModel(reRead.transform(data))
  }
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

  lazy val _makeRes: UserDefinedFunction = udf { x: Vector =>
//    val curr = x.toArray
    val coefs = DenseVector(_coefs.toArray)

    sum(x.asBreeze.toDenseVector * coefs(0 until coefs.length))
  }
  lazy val _coefs: Vector = Vectors.dense(1.5, 0.3, -0.7)
  lazy val _dataset: Matrix = Matrices.dense(100000, 3, DenseMatrix.rand[Double](100000, 3).toArray)
  lazy val _res: DenseVector[Double] = DenseVector(_dataset.multiply(_coefs.toDense).toArray)
}
