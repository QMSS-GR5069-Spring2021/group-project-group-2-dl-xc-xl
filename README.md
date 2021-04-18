# Group Project - Group 2

Term: Spring 2021

Group 2
## Team Contribution

Derek Li: Q1


Xinyang Chen: Q2


Xuejing Li: Q3 


## Repo Map

### src/
	├── data
	├── features
	├── models
	├── visualizations
### reports/
	├── documents
	├── figures
### references/


## Project Description: 

This project uses inferential model and prediction model to explain and predict the performace of drivers and constructors in the F1 race bewteen 1950 and 2010. The raw data and the code for preprocessing the data can be found under the "src/data" folder. We stored the code for feature building under the "src/features" folder and the code for model experimentations under the "src/models" folder. The "src/visualizations" folder contains screenshots of the model experiments. Under the reports folder, you can find our analysis for all three questions below. Please see each subfolder for a README file.


## Framework of Solution

### Question 1： 

your first task is inferential. You are going to try to explain why a driver arrives in second place in a race between 1950 and 2010. Fit a model using features that make theoretical sense to describe F1 racing between 1950 and 2010. Clean the data, and transform it as necessary, including dealing with missing data. [Remember, this will almost necessarily be an overfit model where variables are selected because they make sense to explain F1 races between 1950 and 2010, and not based on algorithmic feature selection]

    Used logistic regression to explain why a driver arrives in second place in a race between 1950 and 2010. Found that starting in the 3rd grid position was the most important factor for increasing the likelihood of a driver arriving in second place.

### Question 2： 

Now we move on to prediction. Fit a model using data from 1950:2010, and predict drivers that come in second place between 2011 and 2017. [Remember, this is a predictive model where variables are selected as the subset that is best at predicting the target variable and not for theoretical reasons. This means that your model should not overfit and most likely be different from the model in (1).]


     Used random forest and logistic regression for prediction and model validation. 


### Question 3： 
This task is inferential. You are going to try to explain why a constructor wins a season between 1950 and 2010. Fit a model using features that make theoretical sense to describe F1 racing between 1950 and 2010. Clean the data, and transform it as necessary, including dealing with missing data. [Remember, this will almost necessarily be an overfit model where variables are selected because they make sense to explain F1 races between 1950 and 2010, and not based on algorithmic feature selection]

    By calculating the average number of wins for every constructor in each season, we transformed the targeted vairable from a binary variable to a continuous variable. Used Linear regression to explain why a constcutor wins a season between 1950 and 2010, and found that the total points for each constructor is the most influential factor that would increase the constructor's odds of wining the season.  
