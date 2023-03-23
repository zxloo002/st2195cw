import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt
from skimpy import skim

pd.set_option('display.max_columns', None) #change to display all columns

airports = pd.read_csv("airports.csv")
planes = pd.read_csv("plane-data.csv")
carriers = pd.read_csv("carriers.csv")
data_2006 = pd.read_csv("2006.csv")
data_2007 = pd.read_csv("2007.csv")

#Join the data for both years and remove duplicates
flight = pd.concat([data_2006, data_2007]).reset_index(drop = True).drop_duplicates()

#replace empty values with None
flight.replace('', None)

flight.info()

# Filter out cancelled and diverted flights
delayed = flight[(flight["Cancelled"] == 0) & (flight["Diverted"] == 0)].copy()

#Question 1
# Create a new column 'DepInterval' based on CRSDepTime
delayed.loc[:, "DepInterval"]  = pd.cut(delayed["CRSDepTime"], 
                                bins=[0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400], 
                                labels=["0000 ~ 0159 hrs", "0200 ~ 0359 hrs", "0400 ~ 0559 hrs", "0600 ~ 0759 hrs", 
                                        "0800 ~ 0959 hrs", "1000 ~ 1159 hrs", "1200 ~ 1359 hrs", "1400 ~ 1559 hrs", 
                                        "1600 ~ 1759 hrs", "1800 ~ 1959 hrs", "2000 ~ 2159 hrs", "2200 ~ 2359 hrs"], 
                                include_lowest=True)

# Create new columns ADelay and DDelay based on ArrDelay and DepDelay
delayed['ADelay'] = delayed.ArrDelay > 0
delayed['DDelay'] = delayed.DepDelay > 0

delayed.info()

# Filter out flights with positive arrival delay and select relevant columns
arrival_delay = delayed[delayed["ArrDelay"] > 0][["Year", "Month", "DayofMonth", "DayOfWeek", 
                                                  "CRSDepTime", "DepTime", "DepDelay", "ArrDelay", "DepInterval"]].copy()

arrival_delay.info()

dailypercent = delayed.groupby('DepInterval').ADelay.mean().reset_index()

sns.barplot(y= 'DepInterval', x= 'ADelay', data=dailypercent, orient = 'h') #add in axis labels

#ax.bar_label(ax.containters[0])
#fig, ax = plt.subplots(figsize=(6, 8))

avg_delay = arrival_delay.groupby('DepInterval').ArrDelay.mean().reset_index()
sns.barplot(x= 'ArrDelay', y= 'DepInterval', data=avg_delay) #add in axis label

#DayOfWeek
weekpercent = delayed.groupby('DayOfWeek').ADelay.mean().reset_index()
sns.barplot(y= 'DayOfWeek', x= 'ADelay', data= weekpercent, orient = 'h')

avg_delay_week = arrival_delay.groupby('DayOfWeek').ArrDelay.mean().reset_index()
sns.barplot(x= 'ArrDelay', y= 'DayOfWeek', data=avg_delay_week) #add in axis label

#Best time of year (months)
monthpercent = delayed.groupby('Month').ADelay.mean().reset_index()
sns.barplot(y= 'Month', x= 'ADelay', data= weekpercent, orient = 'h')

avg_delay_month = arrival_delay.groupby('Month').ArrDelay.mean().reset_index()
sns.barplot(x= 'ArrDelay', y= 'Month', data=avg_delay_month) #add in axis label

#question 2
plane_data = flight.merge(planes, left_on=('TailNum'), right_on=('tailnum'), how= 'left', indicator=True).copy()
