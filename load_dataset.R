source('helper.R')

cat("Loading Dataset...")
data = load_dataset('/home/2021/nyu/fall/acb9244/dataset/AirlinesCodrnaAdult.csv')	# Load Data

# Missing values are denoted by '?', we replace them by NA
data[data == '?'] <- NA

# Change datatype of Delay from Integer to Categorical
data$Delay <- as.factor(data$Delay)
