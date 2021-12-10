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
* Jupyter Notebook (Data cleaning and Machine Learning Codes)
* PG Admin (SQL DATABASE)

## DATA COLLECTION
Data was obtained from kaggle as a csv file. This was the only viable data available on this topic. 
There was other data available, however, we  decided to choose the data for UK and Ireland.

The csv file included 42 columns and over 744 thousand rows of data initially. The data was cleaned with Pandas on Jupyter Notebook. Google colab was considered as a choice of Juypter notebook, as the local machine would not support the size of the CSV file data, however, there were a lot of limitations with this application and it ran out of free space so the Jupyter Notebook route was followed. 

The csv file included horse racing history between 2005 and 2019. Data was made up of the horse winning history, track conditions, distance, horse parentage, horse weight, age, racing group, time of the year/season the races took place etc 

DEFINITIONS OF THE INDEPENDENT VARIABLES EMPLOYED IN THE MODELS 
Race ID: The Id given during the race
Date: The date the race took place
Course: The course/place where the race is taking place
Time: The time of the race
Race_name: The name given to the race
Class: Grouping of horses based on official ratings
Band: 
Dist.f.: Distance in Furlongs
Dist.m.: Distance in Metres
Going: The condition of the course track
Season: The season of the race
Race_group: Race group
Race_type: Race Type
Month: The month of the year that the race took place
Year: The year the race took place
Period: Period of the horse racing calendar
Runners: Total Number of the horses racing
Race_money: Total money allocated for the race
Horse_name: The name of the horse racing
Trainer: The name of the horse trainer
Jockey: The horse jockey
Pos: Position of the horse at the end of the race
Btn: The amount of time the horse has lost the race
Sp:
Dec:
Age: The age of the horse
Weight: The weight of the horse in kg
Lbs: The weight of the horse in pounds
Gear: The blinkers, nose rolls, and bar plates
Fin_time: Finish Time: the time the horse took to finish the race
Or: Official Rating
Ts: Top Speed
Rpr (Racing Post Rating): Horse rating by Racing Post magazine
Sire: The father of the horse
Dam: The mother of the horse
Damsire:  The horse’s grandfather on the mother's side 
Comments: Comments made after the race
Dec_clean:
Probability: The calculated chance of the horse winning the race
Exp_chance: Expected chance of a horse winning
Act_score:  Score after racing
Prize_money: Money awarded to the horses based on their positions

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



![4-Conversion_to_numerical.PNG](Images/4-Conversion_to_numerical.PNG)



The final cleaned dataframe was converted and exported as a CSV file. This CSV file serves as the source to the SQL database.



![5-Export_to_CSV](Images/5-Export_to_CSV.PNG)


## SQL DATABASE

The cleaned CSV file was imported into the SQL DATABASE, which was created on PG ADMIN.

![6](Images/6-SQL-DB.PNG)

A connection to the SQL database was made on a new Jupyter notebook (Machine-learning) as this would serve as the resource for the data that will be processed in the machine learning element of this project.

![7](Images/7-SQLConn.PNG)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

## MACHINE LEARNING DATA PROCESSING

Jupyter notebook was utilised to further prepare the data and the following Dependencies were imported to work on the machine learning element of this project.



![8](Images/8-Dependencies-for-ML.PNG)


The machine learning data prepared above was imported into this notebook.


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

