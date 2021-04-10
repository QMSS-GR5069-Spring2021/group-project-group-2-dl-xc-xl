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

fig = sm.graphics.plot_ccpr(lg1, "points_x")
fig.tight_layout(pad=1.0)

# COMMAND ----------

fig = sm.graphics.plot_ccpr(lg1, "points_x")
fig.tight_layout(pad=1.0)