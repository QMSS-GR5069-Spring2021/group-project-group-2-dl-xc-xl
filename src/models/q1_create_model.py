# Databricks notebook source
# MAGIC %md
# MAGIC # Create Model for Q1 - Group Project

# COMMAND ----------

# Import modules
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.classification import LogisticRegression

# Read in dataset with features
q1_data = spark.read.csv("s3://group2-gr5069/processed/project/combo_f1_data_w_features.csv", header=True, inferSchema=True)

# Restrict data to 1950 - 2010 only
q1_data = q1_data.filter((F.col("race_year") >= 1950) & (F.col("race_year") <= 2010))

# Create feature vector
feature_cols = [
  "age_as_of_race", "ctor_ferrari", "ctor_mclaren", 
  "ctor_williams", "ctor_team_lotus", "grid_1", 
  "grid_2", "grid_3", "grid_4", "grid_5", "grid_6", "grid_7", 
  "grid_8", "grid_9", "grid_10"
]
vecAssembler = VectorAssembler(inputCols = feature_cols, outputCol = "features")
vec_df = vecAssembler.transform(q1_data)

# Pyspark GLR and logisticRegression models give same results but have different methods/attributes

# GLR model for nice model summary (coefficients, p-values, std.errors, etc)
glr = GeneralizedLinearRegression(featuresCol="features", labelCol="second_place", family="binomial", link="logit")
glr_model = glr.fit(vec_df)

# Logistic regression model for actual predictions
logit = LogisticRegression(featuresCol="features", labelCol="second_place")
logit_model = logit.fit(vec_df)

# Save model summary
summary = glr_model.summary

feature_names = ["intercept"] + feature_cols
coefs = [glr_model.intercept] + [float(num) for num in glr_model.coefficients]
model_stats = list(zip(feature_names, coefs, summary.coefficientStandardErrors, summary.tValues, summary.pValues))

columns = ["feature_name", "coef", "std_error", "t_value", "p_value"]
model_stats_df = spark.createDataFrame(data=model_stats, schema=columns)

model_stats_df.write.csv("s3://group2-gr5069/processed/q1/q1_model_summary.csv", header="true", mode="overwrite")

# Add predictions
df_w_preds = logit_model.transform(vec_df)

# Save dataset with predictions
df_w_preds = df_w_preds.drop("features", "rawPrediction", "probability")
df_w_preds.write.csv("s3://group2-gr5069/processed/q1/q1_preds.csv", header="true", mode="overwrite")