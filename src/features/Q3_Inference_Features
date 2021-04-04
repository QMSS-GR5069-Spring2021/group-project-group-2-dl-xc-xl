# Databricks notebook source
# MAGIC %md #### Group2_Xuejing Li

# COMMAND ----------

# MAGIC %md ### Final Project
# MAGIC (3) [25pts] This task is inferential. You are going to try to explain why a constructor wins a season between 1950 and 2010. Fit a
# MAGIC model using features that make theoretical sense to describe F1 racing between 1950 and 2010. Clean the data, and transform it
# MAGIC as necessary, including dealing with missing data. [Remember, this will almost necessarily be an overfit model where variables
# MAGIC are selected because they make sense to explain F1 races between 1950 and 2010, and not based on algorithmic feature selection]
# MAGIC From your fitted model:
# MAGIC 
# MAGIC ● describe your model, and explain why each feature was selected
# MAGIC 
# MAGIC ● provide statistics that show how well the model fits the data
# MAGIC 
# MAGIC ● what is the most important variable in your model? How did you determine that?
# MAGIC 
# MAGIC ● provide some marginal effects for the variable that you identified as the most important in the model, and interpret it in
# MAGIC the context of F1 races: in other words, give us the story that the data is providing you about constructors that win
# MAGIC seasons
# MAGIC 
# MAGIC ● does it make sense to think of it as an "explanation" for why a constructor wins a season? or is it simply an association
# MAGIC we observe in the data?

# COMMAND ----------

import pandas as pd

# COMMAND ----------

f1 = spark.read.csv('s3://group2-gr5069/processed/project/combo_f1_data_w_features.csv/', header = True)
status = spark.read.csv('s3://columbia-gr5069-main/raw/driver_standings.csv', header=True)

# COMMAND ----------

#Convert PySpark dataframe to pandas dataframe 
f1_pandas = f1.toPandas()
status_pandas = status.toPandas()

# COMMAND ----------

#Get null values and datatype
f1_pandas.info()
status_pandas.info()

# COMMAND ----------

f1_pandas.head()

# COMMAND ----------

status_pandas.head()

# COMMAND ----------

#join tables on RaceId and DriverId 
win = pd.merge(f1_pandas, status_pandas, how="left", on=["raceId", "driverId"])
win.head(10)

# COMMAND ----------

# MAGIC %md ###To Do:
# MAGIC ##### - Pick relevant variables for modeling
# MAGIC ##### - Transform data types
# MAGIC ##### - Look at Summary Statistics
# MAGIC 
# MAGIC #print(f1_pandas[["grid","positionOrder","laps","milliseconds","fastestLap","fastestLapSpeed","rank"]].astype(int).describe())

# COMMAND ----------

