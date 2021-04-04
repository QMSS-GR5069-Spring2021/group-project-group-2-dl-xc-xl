# Databricks notebook source
# MAGIC %md
# MAGIC # Build Features for Q1 - Group Project

# COMMAND ----------

# Import modules
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.types import IntegerType

# Read in q1 dataset
project_data = spark.read.csv("s3://group2-gr5069/processed/project/combo_f1_data.csv", header=True, inferSchema=True)

# Create binary variable to indicate 2nd place finish
project_data = project_data.withColumn('second_place', F.when(F.col("positionOrder") == 2, 1).otherwise(0))

# Create age as of race variable
project_data = project_data.withColumn("age_as_of_race", F.datediff(F.col("race_date"), F.col("dob"))/365)

# Create binary variables for constructors with top 5 most wins (excluding mercedes since they don't have that much relevant data)
project_data = (
  project_data
  .withColumn("ctor_ferrari", F.when(F.col("constructorRef") == "ferrari", 1).otherwise(0))
  .withColumn("ctor_mclaren", F.when(F.col("constructorRef") == "mclaren", 1).otherwise(0))
  .withColumn("ctor_williams", F.when(F.col("constructorRef") == "williams", 1).otherwise(0))
  .withColumn("ctor_team_lotus", F.when(F.col("constructorRef") == "team_lotus", 1).otherwise(0))
)

# Remove grid = 0 records
project_data = project_data.filter(F.col("grid") != 0)

# Create binary variables for top 10 grid positions
project_data = (
  project_data
  .withColumn("grid_1", F.when(F.col("grid") == 1, 1).otherwise(0))
  .withColumn("grid_2", F.when(F.col("grid") == 2, 1).otherwise(0))
  .withColumn("grid_3", F.when(F.col("grid") == 3, 1).otherwise(0))
  .withColumn("grid_4", F.when(F.col("grid") == 4, 1).otherwise(0))
  .withColumn("grid_5", F.when(F.col("grid") == 5, 1).otherwise(0))
  .withColumn("grid_6", F.when(F.col("grid") == 6, 1).otherwise(0))
  .withColumn("grid_7", F.when(F.col("grid") == 7, 1).otherwise(0))
  .withColumn("grid_8", F.when(F.col("grid") == 8, 1).otherwise(0))
  .withColumn("grid_9", F.when(F.col("grid") == 9, 1).otherwise(0))
  .withColumn("grid_10", F.when(F.col("grid") == 10, 1).otherwise(0))
)


# turn a few fields into numeric data types 

project_data = project_data.withColumn("milliseconds", F.col("milliseconds").astype(IntegerType()))
project_data = project_data.withColumn("fastestLap", F.col("fastestLap").astype(IntegerType()))
project_data = project_data.withColumn("rank", F.col("rank").astype(IntegerType()))
project_data = project_data.withColumn("fastestLapSpeed", F.col("fastestLapSpeed").astype(IntegerType()))

# Save
project_data.write.csv("s3://group2-gr5069/processed/project/combo_f1_data_w_features.csv", header="true", mode="overwrite")
