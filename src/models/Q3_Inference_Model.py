# Databricks notebook source
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

import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np

# COMMAND ----------

#Read data csv

constructor_seasonf = spark.read.csv('s3://group2-gr5069/processed/Q3/ConSeason_FINAL.csv/', header = True,inferSchema=True)

#Convert PySpark dataframe to pandas dataframe 
constructor_pd = constructor_seasonf.toPandas()

# COMMAND ----------

cons_pd = constructor_pd.drop(columns=['constructorRef','constructor_nationality'])
cons_pd.info()

# COMMAND ----------

#groupby constructorId, constructorRef, race_year
congroup = cons_pd.groupby(['race_year','constructorId']).mean()

congroup

# COMMAND ----------

# MAGIC %md ####Let's try our first LR model!

# COMMAND ----------

lg1 = smf.ols('wins ~ grid+positionOrder+points_x+laps+age_as_of_race+s_autumn+s_spring+s_summer+s_winter+continent', data = congroup).fit()
print(lg1.summary())

# COMMAND ----------

import matplotlib.pyplot as plt
plt.rc('figure', figsize=(12, 7))
#plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
plt.text(0.01, 0.05, str(lg1.summary()), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
plt.axis('off')
plt.tight_layout()
plt.savefig('output.png')

# COMMAND ----------

#save summary result into dataframe
def results_summary_to_dataframe(results):
    '''take the result of an statsmodel results table and transforms it into a dataframe'''
    pvals = results.pvalues
    coeff = results.params
    conf_lower = results.conf_int()[0]
    conf_higher = results.conf_int()[1]

    results_df = pd.DataFrame({"pvals":pvals,
                               "coeff":coeff,
                               "conf_lower":conf_lower,
                               "conf_higher":conf_higher
                                })

    #Reordering...
    results_df = results_df[["coeff","pvals","conf_lower","conf_higher"]]
    return results_df

# COMMAND ----------

#print summary dataframe of logistic regression
result = results_summary_to_dataframe(lg1)
result

# COMMAND ----------

# MAGIC %md #### Model#1

# COMMAND ----------


#select columns for building the model
X = congroup.loc[:, ['grid','positionOrder', 'points_x','age_as_of_race','s_autumn','s_spring','s_summer','s_winter']]
y = congroup['wins']
X.shape, y.shape

#perform train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
print(X_train.shape)
print(y_train.shape)
np.all(np.isnan(X))

X = X.reset_index()

#Fill all missing values with the mean value of each column
X = X.fillna(X.mean())

#taking log of the column
X = X.round(2)


X


# COMMAND ----------

#install mlflow library
dbutils.library.installPyPI("mlflow", "1.14.0")

# COMMAND ----------

import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet
import sys
from math import sqrt

alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
  
with mlflow.start_run():
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(X_train, y_train)

    predictions = lr.predict(X_test)

    #(rmse, mae, r2) = eval_metrics(y_test, predicted_qualities)

    #print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
    #print("  RMSE: %s" % rmse)
    #print("  MAE: %s" % mae)
    #print("  R2: %s" % r2)
    
    # Create metrics
    rmse = sqrt(mean_squared_error(y_test, predictions))
    rmse_train = sqrt(mean_squared_error(y_train, lr.predict(X_train)))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print("  rmse: {}".format(rmse))
    print("  rmse_train: {}".format(rmse_train))
    print("  mae: {}".format(mae))
    print("  R2: {}".format(r2))
    
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)

    mlflow.sklearn.log_model(lr, "model")

# COMMAND ----------

# MAGIC %md #### Model#2

# COMMAND ----------


#select columns for building the model
X2 = congroup.loc[:, ['points_x']]
y2 = congroup['wins']

#perform train test split
from sklearn.model_selection import train_test_split
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size = 0.2, random_state = 42)
print(X2_train.shape)
print(y2_train.shape)

X2 = X2.reset_index()

#Fill all missing values with the mean value of each column
X2 = X2.fillna(X2.mean())

X2


# COMMAND ----------

alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
  
with mlflow.start_run():
    lr2 = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr2.fit(X2_train, y2_train)

    predictions2 = lr2.predict(X2_test)

    #(rmse, mae, r2) = eval_metrics(y_test, predicted_qualities)

    #print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
    #print("  RMSE: %s" % rmse)
    #print("  MAE: %s" % mae)
    #print("  R2: %s" % r2)
    
    # Create metrics
    rmse = sqrt(mean_squared_error(y2_test, predictions2))
    rmse_train = sqrt(mean_squared_error(y2_train, lr2.predict(X2_train)))
    mae = mean_absolute_error(y2_test, predictions2)
    r2 = r2_score(y2_test, predictions2)
    print("  rmse: {}".format(rmse))
    print("  rmse_train: {}".format(rmse_train))
    print("  mae: {}".format(mae))
    print("  R2: {}".format(r2))
    
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)

    mlflow.sklearn.log_model(lr2, "model")

# COMMAND ----------

