from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnan, count, mean
import pandas as pd
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
spark = SparkSession.builder.appName("Transportation_Analytics_PySpark").getOrCreate()

file_path = '/Users/sahilsingh/Desktop/Spark Project/Transportation data.xlsx'
data = pd.read_excel(file_path)
csv_path = '/Users/sahilsingh/Desktop/Spark Project/Transportation_Data.csv'
data.to_csv(csv_path, index=False)
df = spark.read.csv(csv_path, header=True, inferSchema=True)
print("Initial Schema:")
df.printSchema()

df.show(5)
df = df.drop("Person_ID")
df = df.na.drop()
df.describe(["Age", "Monthly_Income", "Travel_Distance", "Travel_Time", "Fuel_Cost", "Satisfactio\
n_Score"]).show()

df.groupBy("City_Type").count().show()
df.groupBy("Preferred_Transport_Mode").count().show()
df.groupBy("Preferred_Transport_Mode").mean("Satisfaction_Score").show()

for col1 in ["Travel_Distance", "Travel_Time", "Fuel_Cost", "Monthly_Income"]:
    corr = df.stat.corr(col1, "Satisfaction_Score")
print(f"Correlation between {col1} and Satisfaction_Score: {corr:.3f}")
categorical_cols = ["Gender", "Occupation", "City_Type", "Vehicle_Ownership",
                    "Preferred_Transport_Mode", "Purpose_of_Travel"]

for col_name in categorical_cols:
    indexer = StringIndexer(inputCol=col_name, outputCol=col_name + "_Index", handleInvalid="keep\
")
df = indexer.fit(df).transform(df)


feature_cols = ["Age", "Monthly_Income", "Travel_Distance", "Travel_Time",
                "Fuel_Cost"] + [col + "_Index" for col in categorical_cols]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
final_df = assembler.transform(df).select("features", "Satisfaction_Score")
train_data, test_data = final_df.randomSplit([0.8, 0.2], seed=42)
lr = LinearRegression(featuresCol="features", labelCol="Satisfaction_Score")
lr_model = lr.fit(train_data)


print("Model Coefficients:", lr_model.coefficients)
print("Intercept:", lr_model.intercept)


predictions = lr_model.transform(test_data)
predictions.select("Satisfaction_Score", "prediction").show(10)

evaluator = RegressionEvaluator(labelCol="Satisfaction_Score", predictionCol="prediction")
rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
print(f"R-squared (R2): {r2:.3f}")
print(f"Mean Absolute Error (MAE): {mae:.3f}")

lr_tuned = LinearRegression(
    featuresCol="features", 
    labelCol="Satisfaction_Score",
    regParam=0.1,       
    elasticNetParam=0.8  
)
lr_model_tuned = lr_tuned.fit(train_data)
pred_tuned = lr_model_tuned.transform(test_data)
rmse_tuned = evaluator.evaluate(pred_tuned, {evaluator.metricName: "rmse"})
r2_tuned = evaluator.evaluate(pred_tuned, {evaluator.metricName: "r2"})
print(f"After Parameter Tuning - RMSE: {rmse_tuned:.3f}, R2: {r2_tuned:.3f}")
accuracy = r2_tuned * 100
print(f"\nApproximate Model Accuracy: {accuracy:.2f}%")

pred_pd = predictions.select("Satisfaction_Score", "prediction").toPandas()
pred_pd.to_csv("/Users/sahilsingh/Desktop/Spark Project/LinearRegression_Results.csv", index=False)
