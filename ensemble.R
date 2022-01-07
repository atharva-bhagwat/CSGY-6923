library(gbm)
library(randomForest)
library(ggplot2)
library(cowsay)

library(foreach)
library(doParallel)
num_cores <- detectCores()
registerDoParallel(num_cores)

source('preprocess_data.R')

# K-fold cross validation

cv <- createFolds(data$Delay, k=10, list=T)

parallel_cv_glm <- function(){
  p_results <- foreach(fold = cv, .combine=rbind) %dopar% {
    
    data.train.cv <- data[-fold,]
    data.test.cv <- data[fold,]
    
    fit <- glm(Delay~., data=data.train.cv, family='binomial')
    y.pred <- predict(fit, newdata=data.test.cv, type='response')
    y.true <- data.test.cv$Delay
    
    data.frame(predicted=as.factor(ifelse(y.pred > 0.5, 1, 0)), actual=y.true, score=y.pred)
  }
  p_conf_mat <- table(p_results$predicted, p_results$actual)
  pprint('Acc:',round(sum(diag(p_conf_mat))/nrow(data),2))
  jpeg('glm_roc.jpg')
  roc_glm <- roc(p_results$actual, p_results$score, plot=TRUE, print.auc=TRUE, col='blue', main='GLM ROC Curve')
  dev.off()
  print(auc(roc_glm))
}
system.time(parallel_cv_glm())

seq_cv_glm <- function(){
  results_temp <- list()
  i=1
  for (fold in cv){
    data.train.cv <- data[-fold,]
    data.test.cv <- data[fold,]
    
    fit <- glm(Delay~., data=data.train.cv, family='binomial')
    y.pred <- predict(fit, newdata=data.test.cv, type='response')
    y.true <- data.test.cv$Delay
    
    results_temp[[i]] = data.frame(predicted=as.factor(ifelse(y.pred > 0.5, 1, 0)), actual=y.true)
    i = i + 1
  }
  s_results = do.call(rbind, results_temp)
  s_conf_mat <- table(s_results$predicted, s_results$actual)
  pprint('Acc:',round(sum(diag(s_conf_mat))/nrow(data),2))
}
system.time(seq_cv_glm())

# Boosting with cross validation

parallel_cv_gbm <- function(){
  p_results <- foreach(fold = cv, .combine=rbind) %dopar% {
    data.train.cv <- data[-fold,]
    data.test.cv <- data[fold,]
    
    data.train.cv$Delay <- as.character(data.train.cv$Delay)
    data.test.cv$Delay <- as.character(data.test.cv$Delay)
    
    fit <- gbm(Delay~., data=data.train.cv, n.trees=1500)
    y.pred <- predict(object=fit, newdata=data.test.cv, type='response')
    y.true <- data.test.cv$Delay
    
    data.frame(predicted=as.factor(ifelse(y.pred > 0.5, 1, 0)), actual=y.true, score=y.pred)
  }
  p_conf_mat <- table(p_results$predicted, p_results$actual)
  pprint('Acc:',round(sum(diag(p_conf_mat))/nrow(data),2))
  jpeg('gbm_roc.jpg')
  roc_gbm <- roc(p_results$actual, p_results$score, plot=TRUE, print.auc=TRUE, col='green', main='GBM ROC Curve')
  dev.off()
  print(auc(roc_gbm))
}
system.time(parallel_cv_gbm())

seq_cv_gbm <- function(){
  results_temp <- list()
  i=1
  for (fold in cv){
    data.train.cv <- data[-fold,]
    data.test.cv <- data[fold,]
    
    data.train.cv$Delay <- as.character(data.train.cv$Delay)
    data.test.cv$Delay <- as.character(data.test.cv$Delay)
    
    fit <- gbm(Delay~., data=data.train.cv, n.trees=1500)
    y.pred <- predict(object=fit, newdata=data.test.cv, type='response')
    y.true <- data.test.cv$Delay
    
    results_temp[[i]] = data.frame(predicted=as.factor(ifelse(y.pred > 0.5, 1, 0)), actual=y.true)
    i = i + 1
  }
  s_results = do.call(rbind, results_temp)
  s_conf_mat <- table(s_results$predicted, s_results$actual)
  pprint('Acc:',round(sum(diag(s_conf_mat))/nrow(data),2))
}
system.time(seq_cv_gbm())

# Random Forest

rf_model <- randomForest(Delay~., data=data.train)

oob.error.data <- data.frame(
  Trees=rep(1:nrow(rf_model$err.rate), times=3),
  Type=rep(c("OOB", "0", "1"), each=nrow(rf_model$err.rate)),
  Error=c(rf_model$err.rate[,"OOB"], 
          rf_model$err.rate[,"0"], 
          rf_model$err.rate[,"1"]))

rf_err <- ggplot(data=oob.error.data, aes(x=Trees, y=Error)) + geom_line(aes(color=Type))
ggsave('rf_err.jpg',rf_err)

rf_pred <- predict(rf_model, newdata=data.test, type='response')

rf_pred_prob <- predict(rf_model, newdata=data.test, type='prob')

rf_roc <- roc(data.test$Delay, rf_pred_prob[,2], plot=TRUE, print.auc=TRUE, col='purple', main='Random Forest ROC')

rf_conf <- confusionMatrix(data.test$Delay, rf_pred, positive='1')
