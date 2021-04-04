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

