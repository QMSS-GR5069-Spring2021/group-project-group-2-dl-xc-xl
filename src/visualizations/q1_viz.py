# Databricks notebook source
# MAGIC %md
# MAGIC # Visualizations for Q1 - Group Project

# COMMAND ----------

# Import modules
import pandas as pd
import matplotlib.pyplot as plt

# Import q1 model summary and q1 data with preds
model_summary = spark.read.csv("s3://group2-gr5069/processed/q1/q1_model_summary.csv", header=True, inferSchema=True)
data_preds = spark.read.csv("s3://group2-gr5069/processed/q1/q1_preds.csv", header=True, inferSchema=True)

# Convert to pandas dataframe
model_summary = model_summary.toPandas()
data_preds = data_preds.toPandas()

# COMMAND ----------

# Visualize coefficients
fig, ax = plt.subplots(figsize=(7, 5))

(model_summary
   .query("feature_name != 'intercept'")
   .sort_values("coef")
   .plot(kind='barh', x="feature_name", y="coef", ax=ax)
)

ax.set(title="Model Coefficients", ylabel="Feature Name", xlabel="Coefficient Value")
ax.legend().set_visible(False)

# COMMAND ----------

# Create summary table
fig, ax = plt.subplots(figsize=(10, 10))

table = ax.table(cellText=model_summary.values, colLabels=model_summary.columns, loc='center')
fig.patch.set_visible(False)
ax.axis('off')
table.set_fontsize(25)
table.scale(4,4)

# COMMAND ----------

# Visualize predicted vs actual
fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharey=True)

(data_preds
   .groupby("prediction")
   .size()
   .reset_index(name="count")
   .plot(kind="bar", x="prediction", y="count", ax=ax0, width=0.2)
)

(data_preds
   .groupby("second_place")
   .size()
   .reset_index(name="count")
   .plot(kind="bar", x="second_place", y="count", ax=ax1)
)

ax0.legend().set_visible(False)
ax1.legend().set_visible(False)

ax0.set(ylabel="Count", xlabel="Predicted Value")
ax1.set(xlabel = "Actual Value")
fig.suptitle("Second Place Finishes: Predicted vs. Actual")