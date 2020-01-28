library(xgboost)

X_train <- read.csv("training_data.csv", header = TRUE)
X_test <- read.csv("test_data.csv", header = TRUE)

y <-  X_train$TRX_COUNT

xgb <- xgboost(data=data.matrix(X_train[,1:6]),
               label = y,
               eta = 0.1,
               max_depth = 15,
               nround = 25,
               subsample = 0.5,
               colsample_bytree = 0.5,
               seed = 1,
               objective = "reg:linear",
)

print(xgb)

training_scores <- predict(xgb, data.matrix(X_train[,1:6]))

# mean absolute error for training data
mean(abs(training_scores - X_train$TRX_COUNT))

# root mean squared error for training data
sqrt(mean((training_scores - X_train$TRX_COUNT)^2))

test_scores <- predict(xgb, data.matrix(X_test))
write.table(test_scores, file = "test_predictions.csv", row.names = FALSE, col.names = FALSE)
