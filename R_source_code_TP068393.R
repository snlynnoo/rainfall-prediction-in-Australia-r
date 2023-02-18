# Rainfall prediction classification in Australia using Machine Learning in R
# https://github.com/snlynnoo/rainfall-prediction-in-Australia-r


#====================== DATA UNDERSTANDING ====================== 

# ~~~~~ Importing data ~~~~~
rainfall = read.csv("weather_aus_sub.csv", header = T)

# ~~~~~ Understanding data ~~~~~
dim(rainfall)
View(rainfall)
str(rainfall)
summary(rainfall)

# ~~~~~ Data Exploration ~~~~~
library(DataExplorer)
plot_histogram(rainfall)
plot_str(rainfall)

#====================== DATA PRE-PROCESSING ====================== 

# ~~~~~ Replacing white space into NA ~~~~~
library(dplyr)
rainfall = mutate_all(rainfall,na_if,"")

# ~~~~~ Checking missing values ~~~~~
colSums(sapply(rainfall,is.na))

library(naniar)
library(ggplot2)
gg_miss_var(rainfall) + labs(y = "missing values")

plot_missing(rainfall)
plot_bar(rainfall)

# ~~~~~  Removing missing values more than 30% in the column ~~~~~ 
# Cloud9am, Cloud3pm, Evaporation, Sunshine 
rainfall$Cloud9am = NULL
rainfall$Cloud3pm = NULL
rainfall$Evaporation = NULL
rainfall$Sunshine = NULL

# ~~~~~ Removing unusable columns ~~~~~
rainfall$Date = NULL
rainfall$Location = NULL

library(psych) 
describe(rainfall)
str(rainfall)

# std < mean ==> no outliers
# Observation => High Skew var.: Rainfall
boxplot(rainfall$Rainfall)
summary(rainfall$Rainfall)

# ~~~~~ Imputing missing values with column mean for numerical data ~~~~~
# Rainfall var. => median imputation (Skewed distribution)
rainfall$Rainfall = ifelse(is.na(rainfall$Rainfall),
                     ave(rainfall$Rainfall, FUN = function(x) median(x, na.rm = TRUE)),
                     rainfall$Rainfall)
# other numerical => mean imputation 
rainfall$WindGustSpeed[is.na(rainfall$WindGustSpeed)] <- round(mean(rainfall$WindGustSpeed, na.rm = TRUE))
rainfall$WindSpeed9am[is.na(rainfall$WindSpeed9am)] <- round(mean(rainfall$WindSpeed9am, na.rm = TRUE))
rainfall$WindSpeed3pm[is.na(rainfall$WindSpeed3pm)] <- round(mean(rainfall$WindSpeed3pm, na.rm = TRUE))
rainfall$Humidity9am[is.na(rainfall$Humidity9am)] <- round(mean(rainfall$Humidity9am, na.rm = TRUE))
rainfall$Humidity3pm[is.na(rainfall$Humidity3pm)] <- round(mean(rainfall$Humidity3pm, na.rm = TRUE))
rainfall = rainfall %>% mutate_if(is.numeric, funs(replace(.,is.na(.), mean(., na.rm = TRUE))))

# ~~~~~ Checking the mode values of categorical variables ~~~~~ 
rainfall %>%
  group_by(rainfall$WindGustDir) %>%
  summarise( Cnt=n(), perc =round(n()/nrow(.)*100,2)) %>%
  arrange(desc(Cnt))
dim(rainfall)

rainfall %>%
  group_by(rainfall$WindDir3pm) %>%
  summarise( Cnt=n(), perc =round(n()/nrow(.)*100,2)) %>%
  arrange(desc(Cnt))
dim(rainfall)

rainfall %>%
  group_by(rainfall$WindDir9am) %>%
  summarise( Cnt=n(), perc =round(n()/nrow(.)*100,2)) %>%
  arrange(desc(Cnt))
dim(rainfall)

rainfall %>%
  group_by(rainfall$RainToday) %>%
  summarise( Cnt=n(), perc =round(n()/nrow(.)*100,2)) %>%
  arrange(desc(Cnt))
dim(rainfall)

# ~~~~~ Imputing categorical variable by mode ~~~~~
rainfall <- rainfall %>% mutate(WindGustDir = replace(WindGustDir,is.na(WindGustDir),"SE"))
rainfall <- rainfall %>% mutate(WindDir3pm = replace(WindDir3pm,is.na(WindDir3pm),"W"))
rainfall <- rainfall %>% mutate(WindDir9am = replace(WindDir9am,is.na(WindDir9am),"N"))
rainfall <- rainfall %>% mutate(RainToday = replace(RainToday,is.na(RainToday),"No"))

# ~~~~~ Removing missing values in RainToday & RainTomorrow (2% only) ~~~~~
rainfall = na.omit(rainfall)
plot_missing(rainfall)

# ~~~~~ Correlation Matrix  ~~~~~
plot_correlation(rainfall, type=c('continuous')) # between num.

# Strongly correlated num. variables ( 0.8 => dropped )
#- Pressure9am ~ Pressure3pm => Keep => Pressure9am
#- MinTemp ~ MaxTemp ~ Temp9am ~ Temp3pm => Keep => MaxTemp

# ~~~~~ Dropping strongly correlated var.~~~~~
rainfall$Pressure3pm<-NULL
rainfall$Temp3pm<-NULL
rainfall$Temp9am<-NULL
rainfall$MinTemp<-NULL
str(rainfall)

# ~~~~~ Encoding the categorical variables ~~~~~ 
rainfall$RainToday = factor(rainfall$RainToday,
                            levels = c('No', 'Yes'),
                            labels = c(0, 1))

rainfall$WindGustDir = factor(rainfall$WindGustDir,
                           levels = c('N', 'NNE', 'NE', 'ENE', 'E', 'ESE',
                                      'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW',
                                      'W', 'WNW', 'NW', 'NNW' ),
                           labels = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                      13, 14, 15, 16))

rainfall$WindDir9am = factor(rainfall$WindDir9am,
                              levels = c('N', 'NNE', 'NE', 'ENE', 'E', 'ESE',
                                         'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW',
                                         'W', 'WNW', 'NW', 'NNW' ),
                              labels = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                         13, 14, 15, 16))

rainfall$WindDir3pm = factor(rainfall$WindDir3pm,
                              levels = c('N', 'NNE', 'NE', 'ENE', 'E', 'ESE',
                                         'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW',
                                         'W', 'WNW', 'NW', 'NNW' ),
                              labels = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                         13, 14, 15, 16))

# ~~~~~ Converting factor to int for encoded columns ~~~~~ 
rainfall$RainToday = as.numeric(as.character(rainfall$RainToday))
rainfall$WindGustDir = as.numeric(as.character(rainfall$WindGustDir))
rainfall$WindDir9am = as.numeric(as.character(rainfall$WindDir9am))
rainfall$WindDir3pm = as.numeric(as.character(rainfall$WindDir3pm))
str(rainfall)
View(rainfall)

# ~~~~~ Converting Target var. into factor ~~~~~
rainfall$RainTomorrow = factor(rainfall$RainTomorrow,
                             levels = c("Yes", "No"),
                             labels = c("Yes", "No"))

# ~~~~~ Normalization using min-max method ~~~~~ 
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))}

rainfall_1 = rainfall[ , -13] # exclude target var.
rainfall.norm = as.data.frame(lapply(rainfall_1, normalize))
rainfall_normalized = cbind(rainfall.norm, rainfall[13])
str(rainfall_normalized)
describe(rainfall)

# ~~~~~ Class balancing ~~~~~
# Before
table(rainfall_normalized$RainTomorrow)
barplot(table(rainfall_normalized$RainTomorrow), main="RainTomorrow (Before CB)", col=c("skyblue","lightgreen"))

library(ROSE)
rainfall_over = ovun.sample(RainTomorrow ~., 
                            rainfall_normalized, method = "over", 
                                 N = 27846, seed=123)$data 
# After
table(rainfall_over$RainTomorrow)
barplot(table(rainfall_over$RainTomorrow), main="RainTomorrow (After CB)", col=c("skyblue","lightgreen"))

# ~~~~~ Stratified sampling ~~~~~ 
library(caTools)
split = sample.split(rainfall_over$RainTomorrow, SplitRatio = 0.8)
training_set = subset(rainfall_over, split == TRUE)
dim(training_set)
test_set = subset(rainfall_over, split == FALSE)
dim(test_set)

#====================== MODEL IMPLEMENTATION ====================== 

# ========== Logistic Regression ==========

# Base model
library(glmnet)
logreg = glm(RainTomorrow~., data=training_set, family="binomial")
logreg$coefficients
summary(logreg)

# ~~~~~ testing the model with test data ~~~~~
y_pred_lr = predict(logreg, test_set[ ,-13] )
y_pred_lr

y_class_lr = ifelse(y_pred_lr > 0.5, "Yes", "No") # threshold -> 0.5
y_class_lr

cm_lr = table(test_set$RainTomorrow, y_class_lr)
cm_lr

# Evaluation matrix
confusionMatrix(cm_lr)
P_lr = precision(table(test_set$RainTomorrow, y_class_lr)); P_lr
R_lr = recall(table(test_set$RainTomorrow, y_class_lr)); R_lr
F1_Score_lr = 2*P*R/(P+R); F1_Score_lr

# ~~~~~ testing the model with train data ~~~~~
y_pred_lr_tr = predict(logreg, training_set[ ,-13] )
y_pred_lr_tr

y_class_lr_tr = ifelse(y_pred_lr_tr > 0.5, "Yes", "No") # threshold -> 0.5
y_class_lr_tr

cm_lr1 = table(training_set$RainTomorrow, y_class_lr_tr)
cm_lr1

# Evaluation matrix
confusionMatrix(cm_lr1)
P_lr_tr = precision(table(test_set$RainTomorrow, y_class_lr)); P_lr_tr
R_lr_tr = recall(table(test_set$RainTomorrow, y_class_lr)); R_lr_tr
F1_Score_lr_tr = 2*P*R/(P+R); F1_Score_lr_tr

# ========== LR with Cross Validation ==========

custom_control = trainControl(method = 'cv', number = 10)

logreg_cv = caret::train(RainTomorrow~., data = training_set, trControl = custom_control,
                         method = 'glm', family="binomial", metric = "Accuracy")
summary(logreg_cv)

# ~~~~~ testing the model with test data ~~~~~
y_pred_lr_cv = predict(logreg_cv, test_set[ ,-13] )
y_pred_lr_cv

cm_lr_cv = table(test_set$RainTomorrow, y_pred_lr_cv)
cm_lr_cv

# Evaluation matrix
confusionMatrix(cm_lr_cv)
P_lr_cv = precision(table(test_set$RainTomorrow, y_pred_lr_cv)); P_lr_cv
R_lr_cv = recall(table(test_set$RainTomorrow, y_pred_lr_cv)); R_lr_cv
F1_Score_lr = 2*P*R/(P+R); F1_Score

# ~~~~~ testing the model with train data ~~~~~
y_pred_lr_cv_tr = predict(logreg_cv, training_set[ ,-13] )
y_pred_lr_cv_tr

cm_lr_cv_tr = table(training_set$RainTomorrow, y_pred_lr_cv_tr)
cm_lr_cv_tr

# Evaluation matrix
confusionMatrix(cm_lr_cv_tr)
P_lr_cv_tr = precision(table(training_set$RainTomorrow, y_pred_lr_cv_tr)); P_lr_cv_tr
R_lr_cv_tr = recall(table(training_set$RainTomorrow, y_pred_lr_cv_tr)); R_lr_cv_tr
F1_Score_lr_cv_tr = 2*P*R/(P+R); F1_Score_lr_cv_tr

# ==================== Naive Bayes Model ===================== 

# Base model
library(e1071)
# ~~~~~ NB Without Smoothing ~~~~~
nb = naiveBayes(x = training_set[-13], y = training_set$RainTomorrow) 
nb 

# ~~~~~ testing the model with test data ~~~~~
y_pred_test = predict(nb, newdata = test_set[-13])
y_pred_test

# confusion matrix between actual & predicted result
cm_test = table(test_set$RainTomorrow, y_pred_test)
cm_test

# Evaluation matrix
library(caret)
confusionMatrix(test_set$RainTomorrow, y_pred_test) # ACC- 0.7192
P = precision(table(test_set$RainTomorrow, y_pred_test)); P
R = recall(table(test_set$RainTomorrow, y_pred_test)); R
F1_Score = 2*P*R/(P+R); F1_Score

# ~~~~~ testing the model with train data ~~~~~
y_pred_train = predict(nb, newdata = training_set[-13])
y_pred_train

# confusion matrix between actual & predicted result
cm_train = table(training_set$RainTomorrow, y_pred_train)
cm_train

# Evaluation matrix
confusionMatrix(training_set$RainTomorrow, y_pred_train) # ACC- 0.7192
P_tr = precision(table(training_set$RainTomorrow, y_pred_train)); P_tr 
R_tr = recall(table(training_set$RainTomorrow, y_pred_train)); R_tr 
F1_Score_tr = 2*P*R/(P+R); F1_Score_tr

# ========== NB Hyper parameter tuning ==========

library(mlr)
getParamSet("classif.naiveBayes") # getting hyper parameter for NB

# ~~~~~ NB with Smoothing ~~~~~
nb_sm = naiveBayes(x = training_set[-13], y = training_set$RainTomorrow, laplace = 1 )
nb_sm

# ~~~~~ testing the model with test data ~~~~~
y_pred_test_sm = predict(nb_sm, newdata = test_set[-13])
y_pred_test_sm

# confusion matrix between actual & predicted result
cm_test_sm = table(test_set$RainTomorrow, y_pred_test_sm) 
cm_test_sm

# Evaluation matrix
confusionMatrix(test_set$RainTomorrow, y_pred_test_sm)
P1 = precision(table(test_set$RainTomorrow, y_pred_test_sm)); P1
R1 = recall(table(test_set$RainTomorrow, y_pred_test_sm)); R1
F1_Score1 = 2*P1*R1/(P1+R1); F1_Score1

# ~~~~~ testing the model with train data ~~~~~
y_pred_train_sm = predict(nb_sm, newdata = training_set[-13])
y_pred_train_sm

# confusion matrix between actual & predicted result
cm_train = table(training_set$RainTomorrow, y_pred_train_sm)
cm_train

# Evaluation matrix
confusionMatrix(training_set$RainTomorrow, y_pred_train_sm) # ACC- 0.7192
P1_tr = precision(table(training_set$RainTomorrow, y_pred_train_sm)); P1_tr 
R1_tr = recall(table(training_set$RainTomorrow, y_pred_train_sm)); R1_tr 
F1_Score1_tr = 2*P*R/(P+R); F1_Score1_tr

# ========== NB with Cross Validation ==========
x = training_set[-13]
y = training_set$RainTomorrow
nb_cv = caret::train(x, y, 'nb', metric="Accuracy", trControl=trainControl(method='cv', number=10, classProbs=T))
nb_cv

# testing the model with test data
y_pred_test_cv = predict(nb_cv, newdata = test_set[-13])
y_pred_test_cv

# confusion matrix between actual & predicted result
cm_test_nb_cv = table(test_set$RainTomorrow, y_pred_test_cv) 
cm_test_nb_cv

# Evaluation matrix
confusionMatrix(test_set$RainTomorrow, y_pred_test_cv)
P2 = precision(table(test_set$RainTomorrow, y_pred_test_cv)); P2
R2 = recall(table(test_set$RainTomorrow, y_pred_test_cv)); R2
F1_Score2 = 2*P1*R1/(P1+R1); F1_Score2

# ~~~~~ testing the model with train data ~~~~~
y_pred_train_cv = predict(nb_cv, newdata = training_set[-13])
y_pred_train_cv

# confusion matrix between actual & predicted result
cm_train = table(training_set$RainTomorrow, y_pred_train_cv)
cm_train

# Evaluation matrix

confusionMatrix(training_set$RainTomorrow, y_pred_train_cv) # ACC- 0.7192
P2_tr = precision(table(training_set$RainTomorrow, y_pred_train_cv)); P2_tr 
R2_tr = recall(table(training_set$RainTomorrow, y_pred_train_cv)); R2_tr 
F1_Score2_tr = 2*P*R/(P+R); F1_Score2_tr

# ==================== Decision Tree Model =====================
library(caret)
library(rpart)
library(rpart.plot)

# ========== DT without HP  ==========
ent_DTree = rpart(RainTomorrow~., training_set, method="class", parms=list(split="information"))
ent_DTree
rpart.plot(ent_DTree)

# ~~~~~ testing the model with test data ~~~~~
y_pred_test_DT = predict(ent_DTree, test_set, type = "class")
y_pred_test_DT

# Confusion Matrix
confusionMatrix(test_set$RainTomorrow, y_pred_test_DT)
P_DT = precision(table(test_set$RainTomorrow, y_pred_test_DT)); P_DT
R_DT = recall(table(test_set$RainTomorrow, y_pred_test_DT)); R_DT
F1_Score_DT = 2*P_DT*R_DT/(P_DT+R_DT); F1_Score_DT

# Feature importance 
varImp(ent_DTree)

# ~~~~~ testing the model with train data ~~~~~
y_pred_train_DT = predict(ent_DTree, training_set, type = "class")
y_pred_train_DT

# confusion matrix between actual & predicted result
cm_train_DT = table(training_set$RainTomorrow, y_pred_train_DT)
cm_train_DT

# Evaluation matrix
confusionMatrix(training_set$RainTomorrow, y_pred_train_DT) # ACC- 0.7192
P_DT_tr = precision(table(training_set$RainTomorrow, y_pred_train_DT)); P_DT_tr 
R_DT_tr = recall(table(training_set$RainTomorrow, y_pred_train_DT)); P_DT_tr 
F1_Score_DT_tr = 2*P*R/(P+R); P_DT_tr

# ========== DT with HP ==========
# HP tuning using grid
custom = trainControl(method = "cv",
                      number = 10, search = 'grid',
                      verboseIter = T)

tune.gridcart = expand.grid(maxdepth = 1:30)
dt_hp = caret::train(RainTomorrow~., training_set, method = 'rpart2', tuneGrid = tune.gridcart,
                     trControl = custom, metric = 'Accuracy')

tune.gridcart = expand.grid(cp = 0:1)
dt_hp = caret::train(RainTomorrow~., training_set, method = 'rpart', tuneGrid = tune.gridcart,
                     trControl = custom, metric = 'Accuracy')

# ~~~~~ Building DT model with grid HP ~~~~~
library(mlr)
getParamSet("classif.rpart")
 
ent_DTree_hp = rpart(RainTomorrow~., training_set, method = "class", minsplit = 1, minbucket = 10, cp = 0, maxdepth = 4)
rpart.plot(ent_DTree_hp)

varImp(ent_DTree_hp)
# testing the model with test data
y_pred_test_DT_hp = predict(ent_DTree_hp, test_set, type = "class")
y_pred_test_DT_hp

# Evaluation Matrix
confusionMatrix(test_set$RainTomorrow, y_pred_test_DT_hp)
P_DT_hp = precision(table(test_set$RainTomorrow, y_pred_test_DT)); P_DT_hp
R_DT_hp = recall(table(test_set$RainTomorrow, y_pred_test_DT)); R_DT_hp
F1_Score_DT_hp = 2*P_DT*R_DT/(P_DT+R_DT); F1_Score_DT_hp

# ~~~~~ testing the model with train data ~~~~~
y_pred_train_DT_hp = predict(ent_DTree_hp, training_set, type = "class")
y_pred_train_DT_hp

# Evaluation matrix
confusionMatrix(training_set$RainTomorrow, y_pred_train_DT_hp) 
P_DT_hp_tr = precision(table(training_set$RainTomorrow, y_pred_train_DT)); P_DT_hp_tr 
R_DT_hp_tr = recall(table(training_set$RainTomorrow, y_pred_train_DT)); R_DT_hp_tr 
F1_Score_DT_hp_tr = 2*P*R/(P+R); F1_Score_DT_hp_tr

# ============================== END ==============================
