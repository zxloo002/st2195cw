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

flight.loc[1] = ''

#replace empty values with None
flight.replace('', None)

# Filter out cancelled and diverted flights
delayed = flight[(flight["Cancelled"] == 0) & (flight["Diverted"] == 0)]

#Question 1
# Create a new column 'DepInterval' based on CRSDepTime
DepInterval = pd.cut(delayed['CRSDepTime'], 
                     bins=[0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400],)

delayed["DepInterval"] = pd.cut(delayed["CRSDepTime"], 
                                bins=[0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400], 
                                labels=["0000 ~ 0159 hrs", "0200 ~ 0359 hrs", "0400 ~ 0559 hrs", "0600 ~ 0759 hrs", 
                                        "0800 ~ 0959 hrs", "1000 ~ 1159 hrs", "1200 ~ 1359 hrs", "1400 ~ 1559 hrs", 
                                        "1600 ~ 1759 hrs", "1800 ~ 1959 hrs", "2000 ~ 2159 hrs", "2200 ~ 2359 hrs"], 
                                include_lowest=True)

# Create new columns ADelay and DDelay based on ArrDelay and DepDelay
delayed["ADelay"] = pd.Series(np.where(delayed["ArrDelay"] > 0, 1, 0)).astype("category")
delayed["DDelay"] = pd.Series(np.where(delayed["DepDelay"] > 0, 1, 0)).astype("category")

# Filter out flights with positive arrival delay and select relevant columns
arrival_delay = delayed[delayed["ArrDelay"] > 0][["Year", "Month", "DayofMonth", "DayOfWeek", 
                                                  "CRSDepTime", "CRSArrTime", "TailNum", 
                                                  "DepDelay", "ArrDelay", "DepInterval"]]

delayed_grouped = delayed.groupby("DepInterval").agg(Percent=("ADelay", lambda x: round(mean(x) * 100, 2)))

plt = delayed_grouped.plot(kind="barh", x="DepInterval", y="Percent", legend=False)
plt.set_title("Probability of delayed flight")
plt.set_xlabel("Percentage")
plt.set_ylabel("")
plt.invert_yaxis()
for i in plt.containers:
    plt.bar_label(i, label=f"{i.get_width()}%", label_type="edge", padding=5)
plt.grid(axis="x")
