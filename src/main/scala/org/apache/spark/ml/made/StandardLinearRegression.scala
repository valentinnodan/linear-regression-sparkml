package org.apache.spark.ml.made

import breeze.linalg.functions.euclideanDistance
import breeze.linalg.{DenseMatrix, DenseVector, sum}
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.{BooleanParam, DoubleParam, IntParam, Param, ParamMap}
import org.apache.spark.ml.param.shared.{HasInputCol, HasLabelCol, HasOutputCol}
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Encoder, Row}

trait StandardLinearRegressionParams extends HasInputCol with HasOutputCol with HasLabelCol {
  def setInputCol(value: String) : this.type = set(inputCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)
  def setLabelCol(value: String): this.type = set(labelCol, value)

  val iter = new IntParam(
    this, "iter","max amount of iterations")

  val tolerance = new DoubleParam(
    this, "tolerance","tolerance")

  val learningRate = new DoubleParam(
    this,
    "learningRate",
    "learning rate for regression"
  )
  def getIter: Double = $(iter)
  def setIter(value: Int) : this.type = set(iter, value)

  def getLearningRate: Double = $(learningRate)
  def setLearningRate(value: Double) : this.type = set(learningRate, value)

  def getTolerance: Double = $(tolerance)
  def setTolerance(value: Double) : this.type = set(tolerance, value)

  setDefault(iter -> 1000, learningRate -> 0.5, tolerance -> 1e-5)
  setDefault(inputCol -> "features")

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getInputCol, new VectorUDT())

    if (schema.fieldNames.contains($(outputCol))) {
      SchemaUtils.checkColumnType(schema, getOutputCol, new VectorUDT())
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getInputCol).copy(name = getOutputCol))
    }
  }
}

class StandardLinearRegression(override val uid: String) extends Estimator[StandardLinearRegressionModel] with StandardLinearRegressionParams
with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("standardScaler"))

  override def fit(dataset: Dataset[_]): StandardLinearRegressionModel = {

    // Used to convert untyped dataframes to datasets with vectors
    implicit val encoder : Encoder[Vector] = ExpressionEncoder()

    val assembler = new VectorAssembler().setInputCols(Array(getInputCol, getLabelCol)).setOutputCol("result")
    val vectors = assembler.transform(dataset).select("result").as[Vector]
//    val vectors: Dataset[Vector] = dataset.select($(inputCol)).as[Vector]

//    print(vectors)

    val iters = getIter
    val rate = getLearningRate
    val tol = getTolerance
    val size: Int = vectors.first().size
    var prevCoefs = DenseVector.fill(size, Double.PositiveInfinity)
    var coefs = DenseVector.fill(size, 0.0)

    var i = 0
    while (i < iters && euclideanDistance(coefs.toDenseVector, prevCoefs.toDenseVector) > tol) {
      i += 1
      val summary = vectors.rdd.mapPartitions((data: Iterator[Vector]) => {
        val result = data.foldLeft(new MultivariateOnlineSummarizer())(
          (summarizer, vector) => {
            val currX = vector.asBreeze(0 until size - 1).toDenseVector
            val currY = currX.dot(coefs)
            summarizer.add(mllib.linalg.Vectors.fromBreeze((currY - vector.asBreeze(-1)) * currX))
          }
        )
        Iterator(result)
      }).reduce(_ merge _)
      prevCoefs = coefs.copy
      coefs = coefs - rate * summary.mean.asBreeze
    }

    copyValues(new StandardLinearRegressionModel(
      coefs)).setParent(this)

//    val Row(row: Row) =  dataset
//      .select(Summarizer.metrics("mean", "std").summary(dataset($(inputCol))))
//      .first()
//
//    copyValues(new standardLinearRegressionModel(row.getAs[Vector](0).toDense, row.getAs[Vector](1).toDense)).setParent(this)
  }

  override def copy(extra: ParamMap): Estimator[StandardLinearRegressionModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

}

object StandardLinearRegression extends DefaultParamsReadable[StandardLinearRegression]

class StandardLinearRegressionModel private[made](override val uid: String,
                                                  val coefs: DenseVector[Double])
  extends Model[StandardLinearRegressionModel] with StandardLinearRegressionParams with MLWritable {


  private[made] def this(coefs: DenseVector[Double]) =
    this(Identifiable.randomUID("standardLinearRegressionModel"), coefs)

  override def copy(extra: ParamMap): StandardLinearRegressionModel = copyValues(
    new StandardLinearRegressionModel(coefs), extra)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val bCoefs = coefs
    val transformUdf = {
      dataset.sqlContext.udf.register(uid + "_transform",
      (x : org.apache.spark.mllib.linalg.DenseVector) => {
        sum(x.asBreeze.toDenseVector * bCoefs(0 until bCoefs.length))
      })
    }
    print(dataset(getInputCol))
    dataset.withColumn(getOutputCol, transformUdf(dataset(this.getInputCol)))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)

      val vectors = Tuple1(coefs.data.asInstanceOf[Vector])

      sqlContext.createDataFrame(Seq(vectors)).write.parquet(path + "/vectors")
    }
  }
}

object StandardLinearRegressionModel extends MLReadable[StandardLinearRegressionModel] {
  override def read: MLReader[StandardLinearRegressionModel] = new MLReader[StandardLinearRegressionModel] {
    override def load(path: String): StandardLinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)

      val vectors = sqlContext.read.parquet(path + "/vectors")

      // Used to convert untyped dataframes to datasets with vectors
      implicit val encoder : Encoder[Vector] = ExpressionEncoder()

      val coefs =  vectors.select(vectors("_1").as[Vector]).first().asBreeze.toDenseVector

      val model = new StandardLinearRegressionModel(coefs)
      metadata.getAndSetParams(model)
      model
    }
  }
}
