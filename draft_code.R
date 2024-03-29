#uses the following packages
library(dplyr)
library(ggplot2)
library(tidyverse)
library(magrittr)
library(moments)

#import data
airports <- read.csv("airports.csv", header = TRUE)
planes <- read.csv("plane-data.csv", header = TRUE)
carriers <- read.csv("carriers.csv", header = TRUE)
data_2006 <- read.csv("2006.csv")
data_2007 <- read.csv("2007.csv")

#combines the two years as one data frame and remove dupes
flight <- rbind(data_2006, data_2007) %>%
  distinct()
  
#remove cancelled/diverted flight data 
delayed <- flight %>% 
  filter(Cancelled == 0 & Diverted == 0)

#group the flights by their CRS departure time
delayed %<>% mutate(DepInterval = cut(CRSDepTime,breaks = c(0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400),
                           labels =  c("0000 ~ 0159 hrs", "0200 ~ 0359 hrs", "0400 ~ 0559 hrs", "0600 ~ 0759 hrs", 
                                       "0800 ~ 0959 hrs", "1000 ~ 1159 hrs", "1200 ~ 1359 hrs", "1400 ~ 1559 hrs",
                                       "1600 ~ 1759 hrs","1800 ~ 1959 hrs", "2000 ~ 2159 hrs", "2200 ~ 2359 hrs"),
                           right = FALSE))

str(delayed)

#Arrival delays only and the columns for data needed.
arrival_delay <- delayed %>%
  filter(ArrDelay > 0) %>%
  select(Year, Month, DayofMonth, DayOfWeek, DepTime, CRSDepTime, ArrTime, CRSArrTime, FlightNum, TailNum, ArrDelay, DepDelay, DepInterval)

#mean depart delay time for each intervals
TimeOfDay <- arrival_delay %>%
  group_by(DepInterval) %>%
  summarize(AvgDelay = round(mean(ArrDelay), 2))

TimeOfDay

#plot a bar graph (time of day)
least_delay_day <-ggplot(TimeOfDay, aes(x= AvgDelay, y=DepInterval)) +
  geom_col(fill= 'pink') +
  labs(title = "Average Delay Duration per Departure Interval", x = "Duration of Delay (Minutes)", y = "") +
  scale_y_discrete(limits = rev) 
least_delay_day + geom_text(aes(label = AvgDelay), hjust = - 0.1 ,position = position_dodge(width = 1), size = 5)

#day of week, maybe change to vertical
day_of_week <- arrival_delay%>%
  mutate(DayOfWeek = factor(x=DayOfWeek, 
                            levels = c(1,2,3,4,5,6,7), 
                            labels = c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))) %>%
  group_by(DayOfWeek) %>%
  summarise(AvgDelay = round(mean(ArrDelay), 2)) %>%
  ggplot(aes(x = AvgDelay, y = DayOfWeek)) +
  geom_col(fill= '#2fccc7') +
  labs(title = "Average Delay Duration Per Day of Week", x = 'Delay Duration (Minutes)', y = '') +
  scale_y_discrete(limits = rev)

day_of_week + geom_text(aes(label = AvgDelay), hjust = - 0.1 ,position = position_dodge(width = 1), size = 5)

#define time of year, quarters
delayed_data_month <- arrival_delay %>% 
  mutate(MonthQuarter = factor(case_when(
    Month  >= 1 & Month <= 3 ~ "Jan - Mar",
    Month >= 4 & Month <= 6 ~ "Apr - Jun",
    Month >= 7 & Month <= 9 ~ "Jul - Sep",
    Month >= 10 & Month <= 12 ~ "Oct - Dec")))

#bar plot for quarterly, check how to increase axis label size
quarter_delay <- delayed_data_month %>%
  group_by(MonthQuarter) %>%
  summarise(AvgDelay = round(mean(ArrDelay), 2)) %>%
  ggplot(aes(x = MonthQuarter, y = AvgDelay)) +
  geom_col(fill = '#4bb2cc') +
  #scale_x_discrete(limits=rev) +
  labs(title = 'Average Delay Duration for each Quarter', x = "", y = "Delay Duration (Minutes)")

quarter_delay + geom_text(aes(label = AvgDelay), vjust = -0.1, position = position_dodge(width = 1), size = 5)

#bar plot for month
monthly_delay <- arrival_delay %>%
  mutate(Month = factor(Month)) %>%
  group_by(Month) %>%
  summarise(AvgDelay = round(mean(ArrDelay),2)) %>%
  ggplot(aes(x= AvgDelay, y=Month)) +
  geom_col(fill= '#d15836') +
  scale_y_discrete(limits=rev) +
  labs(title = "Average Delay Duration per Month", x = "Delay Duration (Minutes)", y = "")

monthly_delay + geom_text(aes(label = AvgDelay), hjust = - 0.1, size = 5)


#---------------Question 2 starts from here--------

#skewness test for ArrDelay
#skewness(f2$ArrDelay)

#IQR for fixing skewness
#q1 <- quantile(delayed$ArrDelay, 0.25)
#q3 <- quantile(delayed$ArrDelay, 0.75)
#iqr <- q3 - q1
#outliers <- delayed$ArrDelay < q1 - 1.5 * iqr | delayed$ArrDelay > q3 + 1.5 *iqr

#delayed %<>% filter(!outliers)
#skewness(delayed$ArrDelay)

#join planes to data (duration of delay)
t2 <-left_join(delayed, planes, by = c("TailNum" = "tailnum")) %>%
  filter(!(DepDelay <= 0 & ArrDelay <= 0)) #filter out rows with no delays

#change all blank values to NA
t2[t2 == ""] <- NA

#change the type from chr to int and drop all rows with NA in 'year' column
t2 %<>% mutate(year = as.integer(year)) %>%
  drop_na(year)

#show only ArrDelay and year column, filter out flights with no year
t3 <- t2 %>% 
  filter(ArrDelay > 0, year != 0) %>%
  select(ArrDelay, year)

t4 <- t3 %>% 
  mutate(year = as.factor(year)) %>%
  group_by(year) %>%
  summarise(AvgDelay = mean(ArrDelay)) %>% #TotalFlights = n() 
  arrange(year)
t4

#bar plot for average 
graph <- ggplot(t4, aes(x = AvgDelay, y = year)) +
  geom_bar(stat = "identity", fill = '#59c44b') +
  xlab("Average Delay") +
  ylab("Year of plane") +
  scale_y_discrete(limits=rev) +
  theme_classic()

graph

#rough working from here on wards
#t3 <- t2 %>% mutate(AgeofPlane = Year - year) %>%
 # drop_na(AgeofPlane) %>%
  #filter(ArrDelay > 0) %>%
  #group_by(AgeofPlane) %>%
#  summarise(AvgDelay = mean(ArrDelay), AgeofPlane) %>%
 # arrange(AgeofPlane)


#intervals <- split(t2, t2$DepInterval)
#interval1 <- select(intervals[[1]], DepDelay, ArrDelay)



#---------Start of Question 3-------

flight$route <- paste(flight$Origin, flight$Dest, sep = "-")

travel <- flight %>%
  group_by(route, Year) %>%
  summarise(TotalFlights = n())

before <- filter(travel, Year == "2006")
before %<>% rename(Before = TotalFlights)

after <- filter(travel, Year == "2007")
after %<>% rename(After = TotalFlights)

difference <- left_join(before, after, by = "route") %>%
  select(route, Before, After)

difference <- replace(difference,is.na(difference), 0)

h6 <- difference %>%
  summarize(Difference = After - Before) %>%
  arrange(desc(Difference))

head(h6) #increased the most
tail(h6) #decreased the most


#another method, by incoming and outgoing difference for each year, for each state
info <- left_join(flight, airports, by = c("Origin" = "iata")) %>%
  left_join(airports, by= c("Dest" = "iata"))

info <- info %>%
  rename('DestState' = 'state.y','OrigState' = 'state.x') %>%
  select(Year, Origin, Dest, OrigState, DestState)

outbound_2006 <- info %>%
  filter(Year == '2006') %>%
  group_by(OrigState) %>%
  summarise(before = n())

outbound_2007 <- info %>%
  filter(Year == "2007") %>%
  group_by(OrigState) %>%
  summarise(after = n())

outbound <- left_join(outbound_2006, outbound_2007, by = "OrigState") %>%
  mutate(OutDiff = after - before)

inbound_2006 <- info %>%
  filter(Year == "2006") %>%
  group_by(DestState) %>%
  summarise(before = n())

inbound_2007 <- info %>%
  filter(Year == '2007') %>%
  group_by(DestState) %>%
  summarise(after = n())

inbound <- left_join(inbound_2006, inbound_2007, by = "DestState") %>%
  mutate(InDiff = after - before)

changes <- left_join(outbound, inbound, by = c("OrigState" = "DestState")) %>%
  select(OrigState, OutDiff, InDiff) %>%
  rename(State = OrigState) %>%
  na.omit(State) %>%
  mutate(State = factor(State))

#change from wide format to long format
changes_long <- changes %>% gather(key = "Travel", value = "Difference", -State)

#plot a stacked bar chart to show the overall changes for all states
#change the legend names
ggplot(changes_long, aes(fill=Travel, y= Difference, x=State)) +
  geom_bar(position = 'stack', stat = 'identity') +
  labs(title = "Changes in Travel Volume for each State", x = "State", y = "Difference between 2006 and 2007") +
  theme_classic()

#bar chart for difference in outbound flight 
changes2 <- changes %>% mutate(Overall = abs(InDiff) - abs(OutDiff)) %>%
  arrange(desc(Overall))

#bar chart for overall difference
ggplot(changes2, aes(x= State, y = Overall)) +
  geom_col(fill = '#4bc471') +
  theme_classic()

#------start of question 4-------

#determine the impact of departure delay on arrival delay
delayed %>%
  filter(LateAircraftDelay > 0) %>%
  group_by(Origin) %>%
  summarise(Total = n()) %>%
  arrange(desc(Total))

m <- delayed %>%
  filter(Origin == "ORD", LateAircraftDelay > 0) %>%
  group_by(TailNum) %>%
  select(Year, Month, DayofMonth, )

m

#-------Caleb's stuff-------------
most_flights <- cascade2 %>%
  count(Origin) %>%
  arrange(desc(n)) %>%
  slice(1) %>%
  pull(Origin)

delay_effect <- cascade2 %>%
  filter(Origin == most_flights) %>%
  group_by(Dest) %>%
  summarize(avg_arr_delay = mean(ArrDelay, na.rm = TRUE),
            avg_dep_delay = mean(DepDelay, na.rm = TRUE),
            count = n()) %>%
  filter(count > 100) %>%
  ungroup()
delay_effect

ggplot(delay_effect, aes(x = avg_dep_delay, y = avg_arr_delay)) +
  geom_point() +
  geom_smooth(method = lm, se = FALSE, color = "seagreen") +
  labs(title = "Impact of Departure Delay on Arrival Delay",
       x = "Average Departure Delay",
       y = "Average Arrival Delay") +
  theme_fivethirtyeight() +
  theme(axis.title = element_text())

delay_cascade <- cascade2 %>%
  semi_join(delay_effect, by = c("Dest" = "Dest")) %>%
  group_by(Origin) %>%
  summarize(avg_delay = mean(DepDelay, na.rm = TRUE)) %>%
  ungroup() 

ggplot(delay_cascade, aes(x = reorder(Origin, avg_delay), y = avg_delay)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(title = "Departure Delay Cascade by Origin Airport",
       x = "Origin Airport",
       y = "Average Departure Delay") +
  theme(plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 90, hjust = 1)) +
  theme_fivethirtyeight() +
  theme(axis.title = element_text())

#-------question 5------
library(mlr3)
library(mlr3learners)
library(mlr3pipelines)
library(mlr3tuning)

ass <- select(flight, Year, Month, DayofMonth, CRSDepTime, CRSArrTime, DepDelay, ArrDelay, Distance, AirTime)
ass$Delayed <- factor(ifelse(ass$ArrDelay > 1 | ass$DepDelay > 1, 0, 1))
ass <- na.omit(ass)

n <- nrow(ass)
train_set <- sample(n, round(0.5*n))
test_set <- setdiff(1:n, train_set)
task <- TaskClassif$new('ass', backend=ass, target = 'Delayed')
task$select(c('Year', 'Month', 'DayofMonth', 'CRSDepTime', 'CRSArrTime', 'DepDelay', 'ArrDelay', 'Distance', 'AirTime'))
measure <- msr('classif.ce')
task

fencoder <- po("encode", method = "treatment",
               affect_columns = selector_type("factor"))
ord_to_int <- po("colapply", applicator = as.integer,
                 affect_columns = selector_type("ordered"))
tuner <- tnr('grid_search')
terminator <- trm('evals', n_evals = 20)

learner_lr <- lrn("classif.log_reg")
gc_lr <- po('imputemean') %>>%
  po(learner_lr)
glrn_lr <- GraphLearner$new(gc_lr)

glrn_lr$train(task, row_ids = train_set)
glrn_lr$predict(task, row_ids = test_set)$score() 

install.packages("xgboost")
learner_gb <- lrn("classif.xgboost")
gc_gb <- po('imputemean') %>>%
  fencoder %>>% ord_to_int %>>%
  po(learner_gb)
glrn_gb <- GraphLearner$new(gc_gb)

glrn_gb$train(task, row_ids = train_set)
glrn_gb$predict(task, row_ids = test_set)$score() 

learner_plr <- lrn('classif.glmnet') 
gc_plr <- po('scale') %>>% 
  fencoder %>>% ord_to_int %>>%
  po('imputemean') %>>%
  po(learner_plr)
glrn_plr <- GraphLearner$new(gc_plr)
tune_lambda <- ParamSet$new (list(
  ParamDbl$new('classif.glmnet.lambda', lower = 0.001, upper = 2)
))

at_plr <- AutoTuner$new(
  learner = glrn_plr,
  resampling = rsmp('cv', folds = 3),
  measure = measure,
  search_space = tune_lambda,
  terminator = terminator,
  tuner = tuner
)
at_plr$train(task, row_ids = train_set)
at_plr$predict(task, row_ids = test_set)$score()

learner_tree <- lrn("classif.rpart")

gc_tree <- po('imputemean') %>>%
  po(learner_tree)
glrn_tree <- GraphLearner$new(gc_tree)

glrn_tree$train(task, row_ids = train_set)
glrn_tree$predict(task, row_ids = test_set)$score() 

learner_rf <- lrn('classif.ranger') 
learner_rf$param_set$values <- list(min.node.size = 4)
gc_rf <- po('scale') %>>%
  po('imputemean') %>>%
  po(learner_rf)
glrn_rf <- GraphLearner$new(gc_rf)
tune_ntrees <- ParamSet$new (list(
  ParamInt$new('classif.ranger.num.trees', lower = 50, upper = 600)
))
at_rf <- AutoTuner$new(
  learner = glrn_rf,
  resampling = rsmp('cv', folds = 3),
  measure = measure,
  search_space = tune_ntrees,
  terminator = terminator,
  tuner = tuner
)
at_rf$train(task, row_ids = train_set)
at_rf$predict(task, row_ids = test_set)$score() 

install.packages("e1071")
learner_svm <- lrn("classif.svm")

gc_svm <- po('imputemean') %>>% 
  fencoder %>>% ord_to_int %>>%
  po(learner_svm)
glrn_svm <- GraphLearner$new(gc_svm)

glrn_svm$train(task, row_ids = train_set)
glrn_svm$predict(task, row_ids = test_set)$score() 

set.seed(1) # for reproducible results
# list of learners
lrn_list <- list(
  glrn_lr,
  glrn_gb,
  at_plr,
  glrn_tree,
  at_rf,
  glrn_svm
)
# set the benchmark design and run the comparisons
bm_design <- benchmark_grid(task = task, resamplings = rsmp('cv', folds = 3), learners = lrn_list)
bmr <- benchmark(bm_design, store_models = TRUE)

library(mlr3viz)
library(ggplot2)
autoplot(bmr) + theme(axis.text.x = element_text(angle = 45, hjust = 1))

bmr$aggregate(measure)