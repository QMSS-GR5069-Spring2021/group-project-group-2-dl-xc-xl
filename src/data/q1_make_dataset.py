# Databricks notebook source
# MAGIC %md
# MAGIC # Make Dataset for Q1 - Group Project

# COMMAND ----------

# Import modules
from pyspark.sql import functions as F
from pyspark.sql.types import *

# Read in drivers data
drivers = spark.read.csv("s3://columbia-gr5069-main/raw/drivers.csv", header=True, inferSchema=True)
# Keep necessary columns
drivers = drivers.select("driverId", "dob", "driverRef", "nationality").withColumnRenamed("nationality", "driver_nationality")

# Read in races data
races = spark.read.csv("s3://columbia-gr5069-main/raw/races.csv", header=True, inferSchema=True)
# Keep necessary columns
races = (
  races
  .select("raceId", "date", "year", "name")
  .withColumnRenamed("date", "race_date")
  .withColumnRenamed("year", "race_year")
  .withColumnRenamed("name", "race_name")
)

# Read in constructors data
constructors = spark.read.csv("s3://columbia-gr5069-main/raw/constructors.csv", header=True, inferSchema=True)
# Keep necessary columns
constructors = (
  constructors
  .select("constructorId", "constructorRef", "nationality")
  .withColumnRenamed("nationality", "constructor_nationality")
)

# Read in results data
results = spark.read.csv("s3://columbia-gr5069-main/raw/results.csv", header=True, inferSchema=True)
# Keep necessary columns
results = results.select("raceId", "driverId", "constructorId", "grid", "positionOrder","points","laps","milliseconds","fastestLap","rank","fastestLapSpeed","statusId")

# Combine data
combo = (
  races
  .join(results, "raceId", how="left")
  .join(drivers, "driverId", how="left")
  .join(constructors, "constructorId", how="left")
)


# Save data in s3
combo.write.csv("s3://group2-gr5069/processed/project/combo_f1_data.csv", header="true", mode="overwrite")
