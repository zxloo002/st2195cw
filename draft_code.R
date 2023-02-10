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

#combines the two years as one data frame
flight_data <- rbind(data_2006, data_2007)

#convert the data frame into a tibble
flight_data %<>% as_tibble()

#remove (cancelled/diverted flight data and) duplicated data
delayed_data <- flight_data %>% 
  #filter(Cancelled == 0 & Diverted == 0) %>%
  distinct()
head(delayed_data)

#fill in blanks with NA
#delayed_data[delayed_data == ""] <- NA

#group the flights by their CRS departure time
delayed_data %<>% 
  mutate(DepInterval = case_when(
  CRSDepTime < 600 ~ "1",
  CRSDepTime >= 600 & CRSDepTime < 1200 ~ "2",
  CRSDepTime >= 1200 & CRSDepTime < 1800 ~ "3",
  CRSDepTime >= 1800 & CRSDepTime < 2359 ~ "4")
  )
delayed_data

#mean depart delay time for each intervals
trial <- delayed_data %>%
  filter(DepDelay > 0) %>%
  group_by(DepInterval) %>%
  summarize(mean_delay = mean(DepDelay))
trial <- trial %>% select(DepInterval = NA)
