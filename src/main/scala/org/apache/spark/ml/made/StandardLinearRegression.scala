package org.apache.spark.ml.made

import breeze.linalg.{DenseMatrix, DenseVector, sum}
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.linalg.{Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.{BooleanParam, Param, ParamMap}
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Encoder, Row}

trait StandardLinearRegressionParams extends HasInputCol with HasOutputCol {
  def setInputCol(value: String) : this.type = set(inputCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)

  val shiftMean = new BooleanParam(
    this, "shiftMean","Whenever to substract mean")
  def isShiftMean : Boolean = $(shiftMean)
  def setShiftMean(value: Boolean) : this.type = set(shiftMean, value)

  setDefault(shiftMean -> true)

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

    val vectors: Dataset[Vector] = dataset.select(dataset($(inputCol)).as[Vector])

    val dim: Int = AttributeGroup.fromStructField((dataset.schema($(inputCol)))).numAttributes.getOrElse(
      vectors.first().size
    )

    val summary = vectors.rdd.mapPartitions((data: Iterator[Vector]) => {
      val result = data.foldLeft(new MultivariateOnlineSummarizer())(
        (summarizer, vector) => summarizer.add(mllib.linalg.Vectors.fromBreeze(vector.asBreeze)))
      Iterator(result)
    }).reduce(_ merge _)

    copyValues(new StandardLinearRegressionModel(
      summary.mean.asML)).setParent(this)

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
      (x : Vector) => {
        sum(x.asBreeze.toDenseVector * bCoefs(1 until bCoefs.length))
      })
    }

    dataset.withColumn($(outputCol), transformUdf(dataset($(inputCol))))
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
