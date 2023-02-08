#uses the following packages
library(dplyr)
library(ggplot2)
library(tidyverse)
library(magrittr)

#import data
airports <- read.csv("airports.csv", header = TRUE)
planes <- read.csv("plane-data.csv", header = TRUE)
carriers <- read.csv("carriers.csv", header = TRUE)
data_2006 <- read.csv("2006.csv")
data_2007 <- read.csv("2007.csv")

#making dataframe for flight data and removing duplicated data
flight_data <- rbind(data_2006, data_2007)

#remove cancelled/diverted flight data, no delays flight and duplicated data
delayed_data <- flight_data %>% 
  filter(Cancelled == 0 & Diverted == 0) %>%
  filter(flight_data$DepDelay > 0 | flight_data$ArrDelay > 0) %>%
  distinct()

#fill in blanks with NA
delayed_data[delayed_data == ""] <- NA

#change the values for flights that landed/departed early
delayed_data$ArrDelay[delayed_data$ArrDelay < 0] <- 0
delayed_data$DepDelay[delayed_data$DepDelay < 0] <- 0
 

#hourly (6) intervals of the flight data
night <- filter(delayed_data, CRSDepTime < 600) 
night %>% summarize(
  arrival_delay = mean(night$ArrDelay),
  departure_delay = mean(night$DepDelay)
)
morning <- filter(delays, CRSDepTime >= 600, CRSDepTime < 1200)
afternoon <- filter(delays, CRSDepTime >= 1200, CRSDepTime < 1800)
evening <- filter(delays, CRSDepTime >= 1800, CRSDepTime < 2359) 
