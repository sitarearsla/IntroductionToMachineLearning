library(AUC)
library(onehot)
library(xgboost)

X_train <- read.csv("hw08_training_data.csv", header = TRUE)
Y_train <- read.csv("hw08_training_label.csv", header = TRUE)
X_test <- read.csv("hw08_test_data.csv", header = TRUE)

encoder <- onehot(X_train, addNA = TRUE, max_levels = Inf)
X_train_d <- predict(encoder, data = X_train)
X_test_d <- predict(encoder, data = X_test)

pred <- matrix(0, nrow = nrow(X_test_d), ncol = ncol(Y_train))
colnames(pred) <- colnames(Y_train)
pred[,1] <- X_test[, 1]

for (outcome in 1:6) {
  y_t <- Y_train[,outcome + 1]
  valid_customers <- which(is.na(y_t) == FALSE)
  y <-Y_train[valid_customers, outcome + 1]
  xgb <- xgboost(data = X_train_d[valid_customers, -1], 
                            label = y, 
                            eta = 0.01, 
                            max_depth = 3,
                            nrounds = 20, 
                            max_delta_step=10,
                            subsample = 0.5,
                            colsample_bytree = 0.4,
                            gamma = 5,
                            objective = "binary:logistic")
  
  training_scores <- predict(xgb, X_train_d[valid_customers, -1])
  # AUC score for training data
  print(auc(roc(predictions = training_scores, labels = as.factor(Y_train[valid_customers, outcome + 1]))))
  test_scores <- predict(xgb, X_test_d[, -1])
  pred[, outcome + 1] <- test_scores
}

write.table(pred, file = "hw08_test_predictions.csv", row.names = FALSE, sep = ",")
