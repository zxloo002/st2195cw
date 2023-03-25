import pandas as pd
import numpy as np
import pdcast as pdc
import seaborn as sns
from matplotlib import pyplot as plt
from skimpy import skim

pd.set_option('display.max_columns', None) #change to display all columns

airports = pd.read_csv("airports.csv", usecols=['iata', 'state','airport'])
planes = pd.read_csv("plane-data.csv", usecols=['tailnum', 'year'])
carriers = pd.read_csv("carriers.csv")
data_2006 = pd.read_csv("2006.csv")
data_2007 = pd.read_csv("2007.csv")

#Join the data for both years and remove duplicates
flight = pd.concat([data_2006, data_2007]).reset_index(drop = True).drop_duplicates()
del(data_2006, data_2007)

#downcast the dataframe
flight = pdc.coerce_df(flight, pdc.infer_schema(flight))

#replace empty values with None
flight.replace('', np.nan, inplace=True)

# Filter out cancelled and diverted flights
flight = flight[(flight["Cancelled"] == 0) & (flight["Diverted"] == 0)]

# Create new columns ADelay and DDelay based on ArrDelay and DepDelay
flight['ADelay'] = flight.ArrDelay > 0
flight['DDelay'] = flight.DepDelay > 0
flight = flight[(flight["Cancelled"] == 0) & (flight["Diverted"] == 0)]

# Create a new column 'DepInterval' based on CRSDepTime
flight['DepInterval']  = pd.cut(flight["CRSDepTime"], 
                                bins=[0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400], 
                                labels=["0000 ~ 0159 hrs", "0200 ~ 0359 hrs", "0400 ~ 0559 hrs", "0600 ~ 0759 hrs", 
                                        "0800 ~ 0959 hrs", "1000 ~ 1159 hrs", "1200 ~ 1359 hrs", "1400 ~ 1559 hrs", 
                                        "1600 ~ 1759 hrs", "1800 ~ 1959 hrs", "2000 ~ 2159 hrs", "2200 ~ 2359 hrs"], 
                                include_lowest=True)

#Question 1

delayed = flight.drop(['TaxiIn', 'TaxiOut','CancellationCode','Diverted', 'Cancelled','UniqueCarrier'], axis=1)

delayed.info()

# Filter out flights with positive arrival delay and select relevant columns
arrival_delay = delayed[delayed["ArrDelay"] > 0][["Year", "Month", "DayofMonth", "DayOfWeek", 
                                                  "CRSDepTime", "DepTime", "DepDelay", "ArrDelay", "DepInterval"]].copy()

arrival_delay.info()


dict = {"1" : 'Monday', "2" : 'Tuesday', "Hadoop": 'H', "Python" : 'P', "Pandas": 'P'}




#question 2
#join planes to flight
plane_data = flight.merge(planes, left_on=('TailNum'), right_on=('tailnum'), how= 'left',).drop('tailnum', axis=1)
plane_data.rename(columns={'year':'ManYear'}, inplace=True)

#Change the data type of 'ManYear' to numeric
plane_data['ManYear'] = pd.to_numeric(plane_data['ManYear'], errors = 'coerce')
plane_data.dropna(subset=['ManYear'],inplace=True)

plane_data = plane_data.query("ManYear != 0")

plane_data = plane_data.astype({"Year": 'int16', "ManYear" : 'int16'})

#create a new column to show age of aircraft
plane_data['age'] = plane_data['Year'] - plane_data['ManYear']

plane_data = plane_data.query("age > 0")

#percentage of flight delay
planepercent = plane_data.groupby('age').apply(lambda x: ((x['ADelay']==1) | (x['DDelay']==1)).mean() * 100).reset_index(name='Percent')
sns.scatterplot(x= 'age', y = 'Percent', data=planepercent)


#Question 3
info = flight.merge(airports, left_on=('Origin'), right_on='iata', how= 'left').drop('iata',axis=1).copy()
info.rename(columns={'state' :'OrigState', 'airport':'OrigAirport'},inplace=True)

info = info.merge(airports, left_on=('Dest'), right_on=('iata'), how = 'left').drop('iata',axis=1)
info.rename(columns={'state' :'DestState', 'airport':'DestAirport'},inplace= True)

info = info[["Year", "Origin", "Dest", "OrigState", "DestState"]]
info = info.astype({'Year': 'int32', 'OrigState':'category', 'DestState':'category'})

#outbound flights
outbound = info.groupby(['OrigState', 'Year']).size().reset_index(name='Total')
outbound = outbound.pivot(index='OrigState', columns='Year', values='Total')

#find the difference
outbound['Diff'] = outbound[2007] - outbound[2006]
outbound = outbound.reset_index().rename(columns={'OrigState': 'State'})

# Remove any rows with missing values and sort the data by the difference
outbound = outbound.dropna().sort_values(by='Diff', ascending=False)

outbound.head(5) #States with the highest increase in outbound flights
outbound.tail(5) #States with the highest decrease in outbound flights

#inbound flights
inbound = info.groupby(['DestState', 'Year']).size().reset_index(name='Total')
inbound = inbound.pivot(index='DestState', columns='Year', values='Total')

#find the difference
inbound['Diff'] = inbound[2007] - inbound[2006]
inbound = inbound.reset_index().rename(columns={'DestState': 'State'})

# Remove any rows with missing values and sort the data by the difference
inbound = inbound.dropna().sort_values(by='Diff', ascending=False)

inbound.head(5) #States with the highest increase in outbound flights
inbound.tail(5) #States with the highest decrease in outbound flights

#bar plots to show
#most and least changes to outbound flight 
mostout = outbound.head(5)
sns.barplot(x='State', y= 'Diff', data=mostout, order=mostout["State"])
leastout = outbound.tail(5)
sns.barplot(x='State', y= 'Diff', data=leastout, order=leastout["State"])

#most and least changes to inbound flight
mostin = inbound.head(5)
sns.barplot(x='State',y= 'Diff',data=mostin, order=mostin["State"])
leastin = inbound.tail(5)
sns.barplot(x='State', y= 'Diff',data=leastin, order=leastin["State"])

#question 4


#question 5
delayed['Delayed'] = ((delayed.ADelay == True) | (delayed.DDelay == True)).astype(int)
skim(delayed)

delayed[['Year', 'Month', 'DayofMonth']] = delayed[['Year', 'Month', 'DayofMonth']].astype('int')
delayed[['DayOfWeek', 'DepInterval']] = delayed[['DayOfWeek', 'DepInterval']].astype('category')

from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split, GridSearchCV      
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.gaussian_process import GaussianProcessClassifier
#from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

delay_model = delayed.sample(n=10**4, random_state=1)
delay_model.shape


features = ['Year', 'Month', 'DayofMonth', 'DayOfWeek', 'DepInterval']
X = delay_model[features].copy()
y = delay_model['Delayed']

numerical_features = ['Year', 'Month', 'DayofMonth']
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler())])

categorical_features = ['DayOfWeek', 'DepInterval']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer()),
    ('onehot', OneHotEncoder(drop=None, handle_unknown='ignore'))])

data_transformer = ColumnTransformer(
    transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features)]) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,random_state=1)



param_grid = {
    'data_transformer__numerical__imputer__strategy': ['mean', 'median'],
    'data_transformer__categorical__imputer__strategy': ['constant','most_frequent']
}

pipe_lr = Pipeline(steps=[('data_transformer', data_transformer),
                      ('pipe_lr', LogisticRegression(max_iter=10000, penalty = None))])
grid_lr = GridSearchCV(pipe_lr, param_grid=param_grid)
grid_lr.fit(X_train, y_train);

pipe_gdb = Pipeline(steps=[('data_transformer', data_transformer),
       ('pipe_gdb',GradientBoostingClassifier(random_state=2))])

grid_gdb = GridSearchCV(pipe_gdb, param_grid=param_grid)
grid_gdb.fit(X_train, y_train);

pipe_plr = Pipeline(steps=[('data_transformer', data_transformer),
                           ('pipe_plr', LogisticRegression(penalty='l1', max_iter=10000, tol=0.01, solver='saga'))])
grid_plr = GridSearchCV(pipe_plr, param_grid=param_grid)
grid_plr.fit(X_train, y_train);

pipe_tree = Pipeline(steps=[('data_transformer', data_transformer),
                           ('pipe_tree', DecisionTreeClassifier(random_state=0))])
grid_tree = GridSearchCV(pipe_tree, param_grid=param_grid)
grid_tree.fit(X_train, y_train);

pipe_rf = Pipeline(steps=[('data_transformer', data_transformer),
                           ('pipe_rf', RandomForestClassifier(random_state=0))])
grid_rf = GridSearchCV(pipe_rf, param_grid=param_grid)
grid_rf.fit(X_train, y_train);

#kernel = 1.0 * RBF(1.0)
#pipe_gp = Pipeline(steps=[('data_transformer', data_transformer),
#                           ('pipe_gp',  GaussianProcessClassifier(kernel=kernel, random_state=0))])
#grid_gp = GridSearchCV(pipe_gp, param_grid=param_grid)
#grid_gp.fit(X_train, y_train);

pipe_svm = Pipeline(steps=[('data_transformer', data_transformer),
                           ('pipe_svm',  LinearSVC(random_state=0, max_iter=10000, tol=0.01))])
grid_svm = GridSearchCV(pipe_svm, param_grid=param_grid)
grid_svm.fit(X_train, y_train);

# create the ROC curve display objects for each classifier
roc_display_lr = RocCurveDisplay.from_estimator(grid_lr, X_test, y_test, name='Logistic Regression')
roc_display_gdb = RocCurveDisplay.from_estimator(grid_gdb, X_test, y_test, name='Gradient Boosting')
roc_display_plr = RocCurveDisplay.from_estimator(grid_plr, X_test, y_test, name='Penalised logistic regression')
roc_display_tree = RocCurveDisplay.from_estimator(grid_tree, X_test, y_test, name='Classification trees')
roc_display_rf = RocCurveDisplay.from_estimator(grid_rf, X_test, y_test, name='Random forests')
roc_display_svm = RocCurveDisplay.from_estimator(grid_svm, X_test, y_test, name='Support vector machines')

# create a new figure and plot the ROC curves on the same axes
fig, ax = plt.subplots(figsize=(8, 6))
roc_display_lr.plot(ax=ax)
roc_display_gdb.plot(ax=ax)
roc_display_plr.plot(ax=ax)
roc_display_tree.plot(ax=ax)
roc_display_rf.plot(ax=ax)
roc_display_svm.plot(ax=ax)

# add the 50-50 line (i.e., diagonal line) to the plot as a reference for a random classifier
ax.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')

# set the x and y limits to be between 0 and 1
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])

# set the plot title and axis labels
ax.set_title('ROC curves for different classifiers')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')

# show the plot
plt.show()
