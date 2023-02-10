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

#making data frame for flight data and removing duplicated data
flight_data <- rbind(data_2006, data_2007)

#remove cancelled/diverted flight data and duplicated data
delayed_data <- flight_data %>% 
  filter(Cancelled == 0 & Diverted == 0) %>%
  distinct()
head(delayed_data)

#fill in blanks with NA
delayed_data[delayed_data == ""] <- NA

#change the values for flights that landed/departed early
delayed_data$ArrDelay[delayed_data$ArrDelay < 0] <- 0
delayed_data$DepDelay[delayed_data$DepDelay < 0] <- 0

#group the flights by their departure time
delayed_data %<>% 
  mutate(interval_deptime = case_when(
  CRSDepTime >= 000 & CRSDepTime < 600 ~ "1",
  CRSDepTime >= 600 & CRSDepTime < 1200 ~ "2",
  CRSDepTime >= 1200 & CRSDepTime < 1800 ~ "3",
  CRSDepTime >= 1800 & CRSDepTime < 2359 ~ "4")
  )
delayed_data

#quarter hourly (6) intervals of the flight data, for departure delays
night <- filter(delayed_data, grepl('Q1', interval_deptime))
delaynig <- night %>% summarize(arrival_delay = mean(night$ArrDelay))

morning <- filter(delayed_data, grepl('Q2', interval_deptime))
delaymor <- morning %>% summarize(arrival_delay = mean(morning$ArrDelay))

afternoon <- filter(delayed_data, grepl('Q3', interval_deptime))
delayaft <- afternoon %>% summarize(arrival_delay = mean(afternoon$ArrDelay))

evening <- filter(delayed_data, grepl('Q4', interval_deptime)) 
delayeve <- evening %>% summarize(arrival_delay = mean(evening$ArrDelay))

test <- delayed_data %>% 
  group_by(interval_deptime)
