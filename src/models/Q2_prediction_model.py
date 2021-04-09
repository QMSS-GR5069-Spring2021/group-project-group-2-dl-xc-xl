# Databricks notebook source
# import mlflow library 

dbutils.library.installPyPI("mlflow", "1.14.0")



# COMMAND ----------

# import libraries and functions 

import boto3
import pandas as pd
import numpy as np
from datetime import datetime

# COMMAND ----------


# Read in dataset
df_combo_spark = spark.read.csv("s3://group2-gr5069/processed/project/combo_f1_data_w_features.csv", header=True, inferSchema=True)


# COMMAND ----------

spark.conf.set("spark.sql.execution.arrow.enabled", "true")

df_combo_orig = df_combo_spark.toPandas()

# COMMAND ----------

display(df_combo_orig)

# COMMAND ----------

df_combo_orig.dtypes

# COMMAND ----------

# drop columns 

df_combo_1 = df_combo_orig.drop(columns = ['race_date','race_name','dob','driverRef','driver_nationality','constructorRef','constructor_nationality','positionOrder','grid'], axis=1)


# COMMAND ----------

df = df_combo_1.dropna()

# COMMAND ----------

display(df)

# COMMAND ----------


train_combo = df.loc[(df['race_year'] >= 1950) & (df['race_year'] <= 2010)]

display(train_combo)

# COMMAND ----------


test_combo = df.loc[(df['race_year'] >= 2011) & (df['race_year'] <= 2017)]

display(test_combo)

# COMMAND ----------

# drop year since we don't need it in our prediction 
#train = train_combo.drop(columns=['race_year'])
#test = test_combo.drop(columns=['race_year'])
train = train_combo
test = test_combo

# subset data to only include position order = 2, since we only need to predict 2nd places 
#train = train_full.loc[train_full['positionOrder'] == 2]
#test = test_full.loc[test_full['positionOrder'] == 2]


# create train and test split here 
X_train = train.loc[:, train.columns != 'second_place']
y_train = train['second_place']
X_test = test.loc[:, test.columns != 'second_place']
y_test = test['second_place']


# COMMAND ----------

# fit a model to see how well the model is at predicting 


# 1. Start an experiment using `mlflow.start_run()` and passing it a name for the run
# 2. Train model
# 3. Log the model using `mlflow.sklearn.log_model()`
# 4. Log the model error using `mlflow.log_metric()`
# 5. Print out the run id using `run.info.run_uuid`

import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

with mlflow.start_run(run_name="Basic RF Experiment") as run:
  # Create model, train it, and create predictions
  rf = RandomForestRegressor()
  rf.fit(X_train, y_train)
  predictions = rf.predict(X_test)
  
  # Log model
  mlflow.sklearn.log_model(rf, "random-forest-model")
  
  # Create metrics
  mse = mean_squared_error(y_test, predictions)
  print("  mse: {}".format(mse))
  
  # Log metrics
  mlflow.log_metric("mse", mse)
  
  # Print out the run id using `run.info.run_uuid`
  runID = run.info.run_uuid
  experimentID = run.info.experiment_id
  
  print("Inside MLflow Run with run_id {} and experiment_id {}".format(runID, experimentID))




# COMMAND ----------

# A function to log Parameters, Metrics, and other Artifacts such as feature importances.
def log_rf(experimentID, run_name, params, X_train, X_test, y_train, y_test):
  import os
  import matplotlib.pyplot as plt
  import mlflow.sklearn
  import seaborn as sns
  from sklearn.ensemble import RandomForestRegressor
  from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
  import tempfile

  with mlflow.start_run(experiment_id=experimentID, run_name=run_name) as run:
    # Create model, train it, and create predictions
    rf = RandomForestRegressor(**params)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)

    # Log model
    mlflow.sklearn.log_model(rf, "random-forest-model")

    # Log params
    [mlflow.log_param(param, value) for param, value in params.items()]
    
    
    # Create metrics
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print("  mse: {}".format(mse))
    print("  mae: {}".format(mae))
    print("  R2: {}".format(r2))

    # Log metrics
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)  
    mlflow.log_metric("r2", r2)  
    
    # Create feature importance
    importance = pd.DataFrame(list(zip(df.columns, rf.feature_importances_)), 
                                columns=["Feature", "Importance"]
                              ).sort_values("Importance", ascending=False)
    
    # Log importances using a temporary file
    temp = tempfile.NamedTemporaryFile(prefix="feature-importance-", suffix=".csv")
    temp_name = temp.name
    try:
      importance.to_csv(temp_name, index=False)
      mlflow.log_artifact(temp_name, "feature-importance.csv")
    finally:
      temp.close() # Delete the temp file
    
    # Create plot
    fig, ax = plt.subplots()

    sns.residplot(predictions, y_test, lowess=True)
    plt.xlabel("Predicted values for Fastest Lap Speed (km/h)")
    plt.ylabel("Residual")
    plt.title("Residual Plot")

    # Log residuals using a temporary file
    temp = tempfile.NamedTemporaryFile(prefix="residuals-", suffix=".png")
    temp_name = temp.name
    try:
      fig.savefig(temp_name)
      mlflow.log_artifact(temp_name, "residuals.png")
    finally:
      temp.close() # Delete the temp file
      
    display(fig)
    return run.info.run_uuid

# COMMAND ----------

# Run other models with new parameters 

params = {
  "n_estimators": 100,
  "max_depth": 5,
  "random_state": 42
}

log_rf(experimentID, "First Run", params, X_train, X_test, y_train, y_test)



params_1000_trees = {
  "n_estimators": 1000,
  "max_depth": 10,
  "random_state": 42
}

log_rf(experimentID, "Second Run", params_1000_trees, X_train, X_test, y_train, y_test)



# COMMAND ----------

# Run more models with new parameters 

params_200_trees = {
  "n_estimators": 200,
  "max_depth": 20,
  "random_state": 42
}

log_rf(experimentID, "3rd Run", params_200_trees, X_train, X_test, y_train, y_test)


# COMMAND ----------

params_300_trees = {
  "n_estimators": 300,
  "max_depth": 20,
  "random_state": 42
}

log_rf(experimentID, "4th Run", params_300_trees, X_train, X_test, y_train, y_test)


# COMMAND ----------

# It seems the above models all pretty good models, and then we choose the last model -- 4th run 
# model, and further analyze its feature importance. 

# From the summary statistics, we can see that "points" are the most important (~0.85) feature among all other features. Therefore
# we first run a regression model on "points" and "second place" to further assess the marginal effect. 

# COMMAND ----------

# Run logistic regression on "points" and "second place" in the model. 

# COMMAND ----------

feature_cols =["points"]

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

vecAssembler = VectorAssembler(inputCols = feature_cols, outputCol = "features")

train_spark = spark.createDataFrame(data=train)
test_spark = spark.createDataFrame(data=test)
df_spark = spark.createDataFrame(data=df)

vecTrainDF = vecAssembler.transform(train_spark)
vecTestDF = vecAssembler.transform(test_spark)
vecDF = vecAssembler.transform(df_spark)

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(labelCol="second_place", featuresCol="features",maxIter=10)

model=lr.fit(vecTrainDF)
predict_train=model.transform(vecTrainDF)
predict_test=model.transform(vecTestDF)
predict_test.select("second_place","prediction").show(10)

# COMMAND ----------

# evaluate the model using BinaryClassificationEvaluator class in Spark ML. BinaryClassificationEvaluator uses areaUnderROC

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator=BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='second_place')

predict_test.select("second_place","prediction","probability").show(5)
print("The area under ROC for train set is {}".format(evaluator.evaluate(predict_train)))
print("The area under ROC for test set is {}".format(evaluator.evaluate(predict_test)))

# COMMAND ----------

# Alternatively, we can run GLR model to get summary statistics on coefficients, p-values, and standard errors. 
from pyspark.ml.regression import GeneralizedLinearRegression

glr = GeneralizedLinearRegression(featuresCol="features", labelCol="second_place")
glr_model = glr.fit(vecDF)

# Save model summary
summary = glr_model.summary


# COMMAND ----------

print(summary)

# COMMAND ----------

# Also run this on all variables 

# COMMAND ----------

df_spark_2 = spark.createDataFrame(data=df)

input_cols_2 = ['constructorId','driverId','raceId','race_year','points','laps','milliseconds','fastestLap','fastestLapSpeed',"age_as_of_race", "ctor_ferrari", "ctor_mclaren", "ctor_williams", "ctor_team_lotus", "grid_1", "grid_2", "grid_3", "grid_4", "grid_5", "grid_6", "grid_7", "grid_8", "grid_9", "grid_10"]

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
# tranform the features columns 

vecAssembler = VectorAssembler(inputCols = input_cols_2, outputCol = "features_2")

vec_df_2 = vecAssembler.transform(df_spark_2)

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
# Run logistic regression model to see model summary and marginal effect 
logit = LogisticRegression(featuresCol="features_2", labelCol="second_place")
logit_model = logit.fit(vec_df_2)

predict_all_2=logit_model.transform(vec_df_2)

# COMMAND ----------


predict_all_2.select("second_place","prediction").show(10)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator=BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='second_place')

predict_all_2.select("second_place","prediction","probability").show(5)
print("The area under ROC for train set is {}".format(evaluator.evaluate(predict_all_2)))

# COMMAND ----------

# Alternatively, we can run GLR model to get summary statistics on coefficients, p-values, and standard errors. 
from pyspark.ml.regression import GeneralizedLinearRegression

glr_2 = GeneralizedLinearRegression(featuresCol="features_2", labelCol="second_place")
glr_model_2 = glr_2.fit(vec_df_2)

# Save model summary
summary2 = glr_model_2.summary

# COMMAND ----------

print("Coefficients: " + str(glr_model_2.coefficients))
print("Intercept: " + str(glr_model_2.intercept))
summary = glr_model_2.summary
print("Dispersion: " + str(summary.dispersion))
print("Null Deviance: " + str(summary.nullDeviance))
print("Residual Degree Of Freedom Null: " + str(summary.residualDegreeOfFreedomNull))
print("Deviance: " + str(summary.deviance))
print("Residual Degree Of Freedom: " + str(summary.residualDegreeOfFreedom))
print("AIC: " + str(summary.aic))
print("Deviance Residuals: ")
summary.residuals().show()

# COMMAND ----------

coefs = [glr_model_2.intercept] + [float(num) for num in glr_model_2.coefficients]

model_stats_df = spark.createDataFrame(data=model_stats, schema=columns)


# COMMAND ----------

display(model_stats_df)

# COMMAND ----------

# Now run a model on all the features 

# COMMAND ----------

# Save model summary
summary = logit_model.summary

coefs = [glr_model.intercept] + [float(num) for num in glr_model.coefficients]
model_stats = list(zip(coefs))

columns = ["coef"]
model_stats_df = spark.createDataFrame(data=model_stats, schema=columns)

# COMMAND ----------

display(model_stats_df)

# COMMAND ----------

 #From Q1, the most important variable is grid_3.
  
  