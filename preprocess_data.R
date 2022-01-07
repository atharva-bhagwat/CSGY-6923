library(sqldf)
library(tidyr)
library(dplyr)
library(caret)

# Importing load_dataset
source('load_dataset.R')

# Feature engineering
data <- data[, !names(data) %in% c('education.num')]
data$education <- recode_factor(data$education,"Preschool"="Dropout","1st-4th"="Dropout","5th-6th"="Dropout","7th-8th"="Dropout","9th"="Dropout","10th"="Dropout","11th"="Dropout","12th"="Dropout","HS-grad"="HighGrad","Some-college"="Community","Assoc-acdm"="Community","Assoc-voc"="Community","Prof-school"="Masters")

# Deleting rows with NA
data <- drop_na(data)

# Convert character/factor features to numeric
data$Airline <- as.numeric(data$Airline)
data$AirportFrom <- as.numeric(data$AirportFrom)
data$AirportTo <- as.numeric(data$AirportTo)
data$education <- as.numeric(data$education)
data$marital.status <- as.numeric(data$marital.status)
data$relationship <- as.numeric(data$relationship)
data$race <- as.numeric(data$race)
data$sex <- as.numeric(data$sex)
data$occupation <- as.numeric(data$occupation)
data$workclass <- as.numeric(data$workclass)
data$native.country <- as.numeric(data$native.country)

# Removing correlated features
data.cor <- cor(data[,-which(names(data)=="Delay")])

cor_index <- which(abs(data.cor)>0.5 & abs(data.cor)!=1, arr.ind = T)
cor_index <- cor_index[!duplicated(cbind(pmax(cor_index[,1], cor_index[,2]), pmin(cor_index[,1], cor_index[,2]))),]
tbl_cor_index <- table(cor_index[,1])
cor_index_num=length(tbl_cor_index)
cor_attributes=as.numeric(names(tbl_cor_index))

data <- data[,-cor_attributes]

# Split dataset
set.seed(42)
data.randomized <- data[sample(1:nrow(data),nrow(data)),]

split_indexes <- sample(1:nrow(data),0.7*nrow(data),replace=F)
data.train <- data.randomized[split_indexes,]
data.test <- data.randomized[-split_indexes,]
