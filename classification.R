library(e1071)
library(rpart)
library(rpart.plot)
library(pROC)

source('preprocess_data.R')

# Logistic Regression (GLM)

glm_model <- glm(Delay~., data=data.train, family='binomial')
pred_train_glm <- predict(glm_model, data.train, type='response')
pred_train_glm_bin <- as.factor(ifelse(pred_train_glm > 0.5, 1, 0))
confusion_mat_train_glm <- confusionMatrix(pred_train_glm_bin, data.train$Delay, positive='1')
pred_test_glm <- predict(glm_model, data.test, type='response')
pred_test_glm_bin <- as.factor(ifelse(pred_test_glm > 0.5, 1, 0))
confusion_mat_test_glm <- confusionMatrix(pred_test_glm_bin, data.test$Delay, positive='1')

pprint('GLM Summary:',summary(glm_model))
pprint('GLM Training Metrics:',confusion_mat_train_glm)
pprint('GLM Testing Metrics:',confusion_mat_test_glm)

roc_glm <- roc(data.test$Delay, pred_test_glm, plot=TRUE, print.auc=TRUE, col='green', main='GLM ROC Curve')

# Naive Bayes

nb_model <- naiveBayes(Delay~., data=data.train)
pred_train_nb <- predict(nb_model, newdata=data.train)
confusion_mat_train_nb <- confusionMatrix(pred_train_nb, data.train$Delay, positive='1')
pred_test_nb_raw <- predict(nb_model, newdata=data.test, type='raw')
pred_test_nb <- predict(nb_model, newdata=data.test)
confusion_mat_test_nb <- confusionMatrix(pred_test_nb, data.test$Delay, positive='1')

pprint('Naive Bayes Training Metrics:',confusion_mat_train_nb)
pprint('Naive Bayes Testing Metrics:',confusion_mat_test_nb)

roc_nb <- roc(data.test$Delay, pred_test_nb_raw[,2], plot=TRUE, print.auc=TRUE, col='red', main='Naive Bayes ROC Curve')

# SVM

svm_model <- svm(Delay~., data=data.train, type='C-classification', kernel='radial')
pred_train_svm <- predict(svm_model, newdata=data.train)
confusion_mat_train_svm <- confusionMatrix(pred_train_svm, data.train$Delay, positive='1')
pred_test_svm <- predict(svm_model, newdata=data.test)
confusion_mat_test_svm <- confusionMatrix(pred_test_svm, data.test$Delay, positive='1')

pprint('SVM Training Metrics:',confusion_mat_train_svm)
pprint('SVM Testing Metrics:',confusion_mat_test_svm)

roc_svm <- roc(data.test$Delay, pred_test_svm, plot=TRUE, print.auc=TRUE, col='cyan', main='SVM ROC Curve')

# Decision Tree

dtree_model <- rpart(Delay~., data=data.train)
pred_train_dtree <- predict(dtree_model, newdata=data.train, type='class')
confusion_mat_train_dtree <- confusionMatrix(pred_train_dtree, data.train$Delay, positive='1')
pred_test_dtree <- predict(dtree_model, newdata=data.test, type='prob')
pred_test_dtree_bin <- predict(dtree_model, newdata=data.test, type='class')
confusion_mat_test_dtree <- confusionMatrix(pred_test_dtree_bin, data.test$Delay, positive='1')

rpart.plot(dtree_model)

pprint('Decision Tree Training Metrics:',confusion_mat_train_dtree)
pprint('Decision Tree Testing Metrics:',confusion_mat_test_dtree)

roc_dtree <- roc(data.test$Delay, pred_test_dtree[,2], plot=TRUE, print.auc=TRUE, col='blue', main='Decision Tree ROC Curve')

# Nearest Neighbor

# As the dataset is too large, ibm cannot load the entire train/test set to run predictions using knn.
# This function takes every 1000 entries, runs predictions and appends to a list. This list is returned at the end containing all the predictions.
predict_knn <- function(data, pred_type){
  i = 1
  final <- c()
  while(i<nrow(data)){
    if (i+999>=nrow(data)){
      cat(i,'/',nrow(data),'\n')
      pred <- predict(knn_model, newdata=data[i:nrow(data),], type=pred_type)
      i = i + 1000
    }
    else{
      cat(i,'/',nrow(data),'\n')
      pred <- predict(knn_model, newdata=data[i:(i+999),], type=pred_type)
      i = i+1000
    }
    if (pred_type=='prob'){
      final <- c(final, as.character(pred[,2]))
    }
    else{
      final <- c(final, as.character(pred))
    }
    
  }
  return (final)
}

knn_model <- gknn(Delay~., data=data.train)
pred_train_knn <- predict_knn(data.train,'class')
confusion_mat_train_knn <- confusionMatrix(pred_train_knn, data.train$Delay, positive='1')
pred_test_knn <- predict_knn(data.test,'class')
pred_test_knn_prob <- predict_knn(data.test,'prob')
confusion_mat_test_knn <- confusionMatrix(as.factor(pred_test_knn), data.test$Delay, positive='1')

pprint('KNN Training Metrics:',confusion_mat_train_knn)
pprint('KNN Testing Metrics:',confusion_mat_test_knn)

roc_knn <- roc(data.test$Delay, as.numeric(pred_test_knn_prob), plot=TRUE, print.auc=TRUE, col='purple', main='KNN ROC Curve')
