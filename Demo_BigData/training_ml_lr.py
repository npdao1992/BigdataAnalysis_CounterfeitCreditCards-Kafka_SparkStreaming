from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, Imputer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Create a SparkSession
spark = SparkSession.builder \
    .appName("LogisticRegressionExample") \
    .getOrCreate()

# Load the data
data = spark.read.csv("onlinefraud.csv", header=True, inferSchema=True)

# Data Cleaning: Remove outliers based on 'amount'
percentiles = data.approxQuantile("amount", [0.25, 0.75], 0.01)
Q1 = percentiles[0]
Q3 = percentiles[1]
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR
lower_bound = Q1 - 1.5 * IQR
data = data.filter((data["amount"] < upper_bound) & (data["amount"] > lower_bound))

# Index the 'type' column
indexer = StringIndexer(inputCol="type", outputCol="type_indexed")
data = indexer.fit(data).transform(data)

# Balance the dataset
class_counts = data.groupBy("isFraud").count().collect()
class_count_0 = int(class_counts[0]["count"])
class_count_1 = int(class_counts[1]["count"])
class_0_ratio = min(class_count_0 / class_count_1, 1.0)
class_0_under = data.filter(data["isFraud"] == 0).sample(False, class_0_ratio)
balanced_data = class_0_under.union(data.filter(data["isFraud"] == 1))

# Create a new feature 'balance_diff'
balanced_data = balanced_data.withColumn("balance_diff", balanced_data["oldbalanceOrg"] - balanced_data["newbalanceOrig"])

# Impute missing values
imputer = Imputer(inputCols=["type_indexed", "amount", "oldbalanceOrg", "newbalanceOrig", "balance_diff"],
                  outputCols=["type_indexed_imputed", "amount_imputed", "oldbalanceOrg_imputed", 
                               "newbalanceOrig_imputed", "balance_diff_imputed"])
imputer_model = imputer.fit(balanced_data)
balanced_data = imputer_model.transform(balanced_data)

# Assemble features into a single vector
assembler = VectorAssembler(inputCols=["type_indexed_imputed", "amount_imputed", "oldbalanceOrg_imputed", 
                                       "newbalanceOrig_imputed", "balance_diff_imputed"], 
                            outputCol="features")
assembler.setHandleInvalid("keep")
balanced_data = assembler.transform(balanced_data)

# Split the data into training and test sets
train_data, test_data = balanced_data.randomSplit([0.8, 0.2], seed=0)

# Train a Logistic Regression model
lr = LogisticRegression(labelCol="isFraud", featuresCol="features", maxIter=10)
lr_model = lr.fit(train_data)

# Make predictions on the test data
predictions = lr_model.transform(test_data)

# Evaluate the model
evaluator = BinaryClassificationEvaluator(labelCol="isFraud", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
accuracy = evaluator.evaluate(predictions)

print("Test AUC = %g" % accuracy)

# Save the model
lr_model.save("lr_model")
