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

s3 = boto3.client('s3')

# Use results table in S3 

bucket = "group2-gr5069"
results_data = "processed/project/combo_f1_data_w_features.csv"

obj_laps = s3.get_object(Bucket= bucket, Key= results_data) 
df = pd.read_csv(obj_laps['Body'])

# COMMAND ----------


# Read in q1 dataset
df_combo_spark = spark.read.csv("s3://group2-gr5069/processed/project/combo_f1_data_w_features.csv", header=True, inferSchema=True)


# COMMAND ----------

spark.conf.set("spark.sql.execution.arrow.enabled", "true")

df_combo_orig = df_combo_spark.toPandas()

# COMMAND ----------

display(df_combo_orig)

# COMMAND ----------

# drop a few non-numeric columns for prediction purpose 



# COMMAND ----------


train_combo = df_combo.filter((F.col("race_year") >= 1950) & (F.col("race_year") <= 2010))
display(train_combo)

# COMMAND ----------

df_combo_orig.dtypes

# COMMAND ----------

# drop columns 

df_combo_1 = df_combo_orig.drop(columns = ['race_date','race_name','dob','driverRef','driver_nationality','constructorRef','constructor_nationality','positionOrder','grid'], axis=1)


# COMMAND ----------

df_combo = df_combo_1.dropna()

# COMMAND ----------

display(df_combo)

# COMMAND ----------


train_combo = df_combo.loc[(df_combo['race_year'] >= 1950) & (df_combo['race_year'] <= 2010)]

display(train_combo)

# COMMAND ----------


test_combo = df_combo.loc[(df_combo['race_year'] >= 2011) & (df_combo['race_year'] <= 2017)]

display(test_combo)

# COMMAND ----------

# drop year since we don't need it in our prediction 
train = train_combo.drop(columns=['race_year'])
test = test_combo.drop(columns=['race_year'])

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



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

