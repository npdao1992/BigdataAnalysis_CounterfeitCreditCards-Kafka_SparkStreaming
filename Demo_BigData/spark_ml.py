from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, when
from pyspark.sql.types import StructType, StringType, DoubleType
from pyspark.ml.feature import VectorAssembler, StringIndexer, Imputer
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StandardScaler

# Initialize the Spark session
spark = SparkSession.builder \
    .appName("KafkaStreamingTransactions") \
    .config("spark.master", "local") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.2") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Define the schema of the JSON data
schema = StructType() \
    .add("step", StringType()) \
    .add("type", StringType()) \
    .add("amount", StringType()) \
    .add("nameOrig", StringType()) \
    .add("oldbalanceOrg", StringType()) \
    .add("newbalanceOrig", StringType()) \
    .add("nameDest", StringType()) \
    .add("oldbalanceDest", StringType()) \
    .add("newbalanceDest", StringType()) \
    .add("isFraud", StringType()) \
    .add("isFlaggedFraud", StringType()) \
    .add("type_numeric", StringType())


transactions_df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9093") \
    .option("subscribe", "messages") \
    .option("startingOffsets", "earliest") \
    .load() \
    .selectExpr("CAST(value AS STRING) as json_string")


parsed_df = transactions_df \
    .select(from_json(col("json_string"), schema).alias("data")) \
    .select("data.*")


type_mapping = when(col("type") == "CASH_OUT", 1) \
    .when(col("type") == "PAYMENT", 2) \
    .when(col("type") == "CASH_IN", 3) \
    .when(col("type") == "TRANSFER", 4) \
    .when(col("type") == "DEBIT", 5) \
    .otherwise(0)  # Handle unexpected types


mapped_df = parsed_df.withColumn("type_numeric", type_mapping.cast("double"))

mapped_df = mapped_df.withColumn("amount", col("amount").cast("double")) \
                     .withColumn("oldbalanceOrg", col("oldbalanceOrg").cast("double")) \
                     .withColumn("newbalanceOrig", col("newbalanceOrig").cast("double"))

typed_df = mapped_df.withColumn("balance_diff", (col("oldbalanceOrg") - col("newbalanceOrig")).cast("double"))

selected_columns = ["type_numeric", "amount", "oldbalanceOrg", "newbalanceOrig", "balance_diff"]


preprocessed_df = typed_df.select(selected_columns)


model_path = "rf_model"
loaded_model = RandomForestClassificationModel.load(model_path)


vector_assembler = VectorAssembler(inputCols=["type_numeric", "amount", "oldbalanceOrg", "newbalanceOrig", "balance_diff"], outputCol="features")
vector_assembler.setHandleInvalid("keep")

streaming_data_with_features = vector_assembler.transform(preprocessed_df)


predictions = loaded_model.transform(streaming_data_with_features)


query = predictions.writeStream \
    .outputMode("append") \
    .format("console") \
    .start()

query.awaitTermination()