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
import numpy as np

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

win.info()

# COMMAND ----------

#Converting selected columns to numeric type
win[["grid", "positionOrder","points_x","laps","milliseconds","rank","fastestLapSpeed","age_as_of_race","position","wins"]] = win[["grid", "positionOrder","points_x","laps","milliseconds","rank","fastestLapSpeed","age_as_of_race","position","wins"]].apply(pd.to_numeric)
win.astype("str")
win['race_date'] =  pd.to_datetime(win['race_date'])

#Add season column
def season_of_date(date):
    year = str(date.year)
    seasons = {'spring': pd.date_range(start='21/03/'+year, end='20/06/'+year),
               'summer': pd.date_range(start='21/06/'+year, end='22/09/'+year),
               'autumn': pd.date_range(start='23/09/'+year, end='20/12/'+year)}
    if date in seasons['spring']:
        return 'spring'
    if date in seasons['summer']:
        return 'summer'
    if date in seasons['autumn']:
        return 'autumn'
    else:
        return 'winter'

# Assuming df has a date column of type `datetime`
win['season'] = win.race_date.map(season_of_date)

#win.groupby("constructorId")["wins"].sum().sort_values(ascending=False)

# COMMAND ----------

win.sort_values(by="race_year")

# COMMAND ----------

#filter for years between 1950 to 2010
wins = win[win["race_year"] <= "2010"]

#select key variables(dropped "rank" and "fatestLapSpeed" because there are too many missing values;didn't include race name becasue we want to explain win by season)
constructor_win = wins[["constructorId","race_year","grid","positionOrder","points_x","laps","constructorRef","constructor_nationality","age_as_of_race","wins","season"]]

#fill missing value with 0
constructor_win["wins"]=constructor_win["wins"].fillna("0")

#convert object to numeirc
constructor_win["wins"] = pd.to_numeric(constructor_win["wins"])
#new dataframe for inference model
constructor_win.info()

# COMMAND ----------

#season can vary in each year
constructor_win[constructor_win["constructorId"] == "1" ]

# COMMAND ----------

#Recode constructor nationality
conditions = [
    (constructor_win["constructor_nationality"].isin(["British","Italian","French","German","Austrian","Swiss","Dutch","Russian","Belgium","Irish","East German"])) ,
    (constructor_win["constructor_nationality"].isin(["American","Canadian","Mexican"])) ,
    (constructor_win["constructor_nationality"].isin(["Malaysian","Japanese","Indian","Hong kong"]))]
choices = [1, 2, 3]
constructor_win['continent'] = np.select(conditions, choices, default=np.nan)

#one-hot-encode season
constructor_win = pd.get_dummies(constructor_win, columns=["season"], prefix = ['s'])


# COMMAND ----------

# MAGIC %md ######Save dataframe to S3

# COMMAND ----------

# Create a Spark DataFrame from a pandas DataFrame using Arrow
constructor_win_sp = spark.createDataFrame(constructor_win)
constructor_win_sp.write.csv('s3://group2-gr5069/processed/Q3/Constructor_season.csv')

# COMMAND ----------

#groupby constructorId, constructorRef, race_year
congroup = constructor_win.groupby(['constructorId','constructorRef','race_year']).mean()
congroup
# Print the first value in each group
#congroup.first()

# COMMAND ----------

# MAGIC %md ####Let's try our first LR model!

# COMMAND ----------

import statsmodels.api as sm
import statsmodels.formula.api as smf

# COMMAND ----------

#groupby constructorId, constructorRef, race_year
congroup = constructor_win.groupby(['constructorId','constructorRef','race_year']).mean()
congroup
# Print the first value in each group
#congroup.first()

# COMMAND ----------

lr1 = smf.ols('wins ~ grid+positionOrder+points_x+laps+age_as_of_race+s_autumn+s_spring+s_summer+s_winter+continent', data = congroup).fit()
print (lr1.summary())

# COMMAND ----------

constructor_results = spark.read.csv('s3://columbia-gr5069-main/raw/constructor_results.csv', header = True)
constructor_standing = spark.read.csv('s3://columbia-gr5069-main/raw/constructor_standings.csv', header=True)
constructor = spark.read.csv('s3://columbia-gr5069-main/raw/constructors.csv', header=True)
season = spark.read.csv('s3://columbia-gr5069-main/raw/seasons.csv', header=True)

# COMMAND ----------

#Convert PySpark dataframe to pandas dataframe 
con_results = constructor_results.toPandas()
con_standing = constructor_standing.toPandas()
con = constructor.toPandas()

# COMMAND ----------

con_standing

# COMMAND ----------

con

# COMMAND ----------

con.nationality.value_counts()

# COMMAND ----------

