# Project 4
## CREATE AN ALGORITHM TO ANALYZE HORSE RACING DATA AND PREDICT FUTURE OUTCOMES
## PROJECT OUTLINE 
The aim is to clean the data, utilizing columns that we need and uploading to a SQL database. We will use machine learning models to create a binary supervised algorithm to make predictions for unseen data in the future. Our aim is to develop a model which combines both fundamental (horse features) and market-generated information (jockey/ratings etc). 

![horseracing.jpg](Images/horseracing.png)


## PREOJECT MEMBERS
### A.Sheikh - C Mafavuke - M Ansah - M Amjad

## INTRODUCTION
Our group has identfied that there is a scope to develop a model which combines both fundamental features of horse racing and market generated information.

## RESOURCES
* Kaggle (CSV File)
* Jupyter Notebook (Coding)
* PG Admin (SQL DATABASE)

## DATA COLLECTION
Data was obtained from kaggle as a csv file. This was the only viable data available on this topic. 
There was other data available, however, we  decided to choose the data for UK and Ireland.

The csv file included 42 columns and over 744 thousand rows of data initially. The data was cleaned with Pandas on Jupyter Notebook. Google colab was considered as a choice of Juypter notebook, as the local machine would not support the size of the CSV file data, however, there were a lot of limitations with this application and it ran out of free space so the Jupyter Notebook route was followed. 

The csv file included horse racing history between 2005 and 2019. Data was made up of the horse winning history, track conditions, distance, horse parentage, horse weight, age, racing group, time of the year/season the races took place etc 

DEFINITIONS OF THE INDEPENDENT VARIABLES EMPLOYED IN THE MODELS 
Independent variable Variable definitions Market-generated 
variable ln ðpsij ÞThe natural logarithm of the normalised final odds probability Fundamental variables 
pre_s_ra Speed rating for the previous race in which the horse ran 
avgsr4 The average of a horse’s speed rating in its last 4 races; 
zero when there is no past run draw 
Post-position in current race eps 
Total prize money earnings (finishing first, second or third) to date/
Number of races entered 
newdis 1 indicates a horse that ran three or four of its last four races at a distance of 80% less than current distance, 0 otherwise 
weight Weight carried by the horse in current race 
win_run The percentage of the races won by the horse in its career 
jnowin The number of wins by the jockey in career to date of race 
jwinper The winning percentage of the jockey in career to date of race 
jst1miss 1 indicates when the other jockey variables are missing; 0 otherwise 
(THE JOURNAL OF PREDICTION MARKETS2007, 1 1)

+++++++++++++++++++++++++++++++++++++++++++++++++++++++

## RAW DATA PROCESSING & CLEANING

Jupyter Notebook was utilised to access the raw data from the CSV files.
The required dependencies were imported:

pandas as pd,

numpy as np,

LogisticRegression,

pyplot as plt,

seaborn as sns,

StandardScaler, 

MinMaxScaler, 

LabelEncoder,

train_test_split,

DecisionTreeClassifier, 

plot_tree,

metrics,

preprocessing,

RandomForestClassifier,

make_classification


A 'for loop' was run to identify the unique categories across the columns.


![1-Initial_DATA_size.PNG](Images/1-Initial_DATA_size.PNG)



A further review of the column data was performed and those features that weren't required in the model such as: 
'race_ID', 'class','time','dist.f.','Month', 'Year', 'Period', 'Runners','fin_time', 'dec','weight','dec_clean','sire', 'dam', 'damsire', 'comment','race_name','btn','sp','or','rpr','horse_name','exp_chance','prob',
were removed. Feature selection was based on elements in the data that contirbuted most to the prediction variable or output that we are interested in. Having irrelevant features in the data could decrease the accuracy of the model, especially the logistic regression that we are using. Our selection of features before modeling the data was to reduce overfitting, improve accuracy and reduce training time.



![2-Dropped_DATA.PNG](Images/2-Dropped_DATA.PNG)



The columns in the dataframe were renamed for ease in understanding their meanings.


![3-Rename_Columns.PNG](Images/3-Rename_Columns.PNG)


Based on the various columns explored, it is clearly necessary to develop a model which combines both fundamental features of each horse and market generated information.

The number of unique values was reduced by binning 'rare' categorical variables on:
1. band
2. jockey
3. trainer
4. track condition  

The raw data size was reduced to under 150K rows over 18 columns.
Further cleaning of the data was performed by reducing categorical variables (band) and non numerical values to a numerical value in the btn (beaten) column. 



![4-Conversion_to_numerical.PNG](Images/4-Conversion_to_numerical.PNG) &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&



The final cleaned dataframe was converted and exported as a CSV file. This CSV file serves as the source to the SQL database.



![5-Export_to_CSV](Images/5-Export_to_CSV.PNG) &&&&&&&&&&&&&&&&&&&&&&&&&&&&&


&&&& D/B screenshots &&&&&&&&&&&&

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

## MACHINE LEARNING DATA PROCESSING

Jupyter notebook was utilised to further prepare the data and the following Dependencies were imported to work on the machine learning element of this project.



![6](Images/6-Dependencies-for-ML.PNG)



The machine learning data prepared above was imported into this notebook.

The 'DATE' column was set as an index for the dataframe.



![7](Images/7-Date-Set-Index.PNG)



Connect prediction with output was performed.


![8](Images/8-Prediction-to-Output.PNG)


X was assigned as a dataframe of the features and y as a series of the outcome variable.



![9](Images/9-X,Y-Features.PNG)


Labels ("Winner") denoted by y were assigned to the train and test data sets.


![10](Images/10-Y-Label.PNG)


'get_dummies' for train data step was performed using:

**X_dum = pd.get_dummies((X_), drop_first=True)
X_dum.head(2)**


sklearn was used to split the dataset and also to split the preprocessed data into a training and testing dataset.

**X_train, X_test, y_train, y_test = train_test_split(X_dum, y_label, train_size = 0.6, random_state=1)**


Next step was to scale and fit the data:

**scaler = StandardScaler()**

**X_scaler = scaler.fit(X_train)**

**X_train_scaled = X_scaler.transform(X_train)**
**X_test_scaled = X_scaler.transform(X_test)**

A Logistic Regression was performed:

**classifier = LogisticRegression(fit_intercept=True, random_state=1, max_iter = 400,verbose=0, multi_class='auto')**
**classifier.fit(X_train_scaled,y_train)**
**classifier.score(X_train_scaled,y_train)**
**classifier.score(X_test_scaled,y_test)**

which returned the following values:

**Training Scaled Data Score: 0.9997335890878091**
**Test  Scaled Data Score: 0.9993606138107417**

