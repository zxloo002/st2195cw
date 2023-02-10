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

#group the flights by their CRS departure time
delayed_data %<>% 
  mutate(DepInterval = case_when(
  CRSDepTime >= 0 & CRSDepTime < 600 ~ "1",
  CRSDepTime >= 600 & CRSDepTime < 1200 ~ "2",
  CRSDepTime >= 1200 & CRSDepTime < 1800 ~ "3",
  CRSDepTime >= 1800 & CRSDepTime < 2400 ~ "4"
  )
)
  
str(delayed_data) #DepInterval is chr type, find a way to change it to int

#mean depart delay time for each intervals
trial <- delayed_data %>%
  filter(DepDelay > 0) %>%
  group_by(DepInterval) %>%
  summarize(mean_delay = mean(DepDelay))
str(trial)

#plot a bar graph (time of day)
least_delay_day <-ggplot(trial, aes(x=mean_delay, y=DepInterval)) +
  geom_col(fill= 'pink') +
  labs(title = "Average Delay Duration per Departure Interval", x = "Duration of Delay Minutes", y = "Departure Interval")
least_delay_day

#day of week
trial2 <- delayed_data %>%
  filter(DepTime >0) %>%
  group_by(DayOfWeek) %>%
  summarise(mean_delay = mean(DepDelay))

#find out how to display day of week as separate bars
least_delay_week <- ggplot(trial2, aes(x=mean_delay, y=DayOfWeek)) +
  geom_col(fill= 'yellow') +
  labs(title = "Average Delay Duration Per Day of Week", x = 'Delay Duration (Minutes)', y = 'Day of Week')
least_delay_week + coord_flip()

#define time of year, quarters
delayed_data_month <- delayed_data %>% 
  mutate(MonthQuarter = case_when(
    Month  >= 1 & Month <= 3 ~ "Quarter 1",
    Month >= 4 & Month <= 6 ~ "Quarter 2",
    Month >= 7 & Month <= 9 ~ "Quarter 3",
    Month >= 10 & Month <= 12 ~ "Quarter 4"
  )
)

trial3 <- delayed_data_month %>%
  filter(DepDelay > 0) %>%
  group_by(MonthQuarter) %>%
  summarise(mean_delay = mean(DepDelay))
str(trial3)

least_delay_months <-ggplot(trial3, aes(x=mean_delay, y=MonthQuarter)) +
  geom_col(fill= 'red') +
  labs(title = "Average Delay Duration per Monthly Quarter", x = "Duration of Delay Minutes", y = "Quarter Month")
least_delay_months


#---------------Question 1 ends here---------

#---------------Question 2 starts from here--------
