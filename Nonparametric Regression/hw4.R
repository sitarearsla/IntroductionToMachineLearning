# read data into memory
data_set <- read.csv("hw04_data_set.csv")

# get waiting and eruption values
w <- data_set$waiting
e <- data_set$eruptions

#the train set
w_train <- w[c(1:150)]
e_train <- e[c(1:150)]

#the test set
w_test <- w[c(151:272)]
e_test <- e[c(151:272)]

#Nonparametric method 1: Regressogram
# get interval
origin  <- 1.5
minimum_value <- max(e)
maximum_value <- min(e)
data_interval <- seq(from = origin, to = maximum_value, by = 0.01)

#bin width and origin
bin_width <- 0.37
N <- length(e_train)

left_borders <- seq(from = origin, to = maximum_value, by = bin_width)
right_borders <- seq(from = origin + bin_width, to = maximum_value+ bin_width, by = bin_width)

#finding p_head
p_head <- sapply(1:length(left_borders), function(b) {sum(left_borders[b] < w_train & w_train <= right_borders[b])}) / (N)

#plotting
plot(e_train, w_train, type = "p", pch = 19, col = "blue", las = 1, main = sprintf("h = %g", bin_width))
points(x=e_test,y=w_test, col = "purple",las = 1,pch = 19)
for (b in 1:length(left_borders)) {
  lines(c(left_borders[b], right_borders[b]), c(p_head[b], p_head[b]), lwd = 2, col = "black")
  if (b < length(left_borders)) {
    lines(c(right_borders[b], right_borders[b]), c(p_head[b], p_head[b + 1]), lwd = 2, col = "black") 
  }
}

init <- 1
N_test <- length(e_test)
merge <- c()
for(t in N_test){
  window <- (e_test[init]-minimum_value)/bin_width
  merge <- cbind(merge, (w_test[init]-p_head[window])^2)
  init <- init + 1
}

#RMSE 
mse <-sum(merge)/N
rmse <- sqrt(mse)
print(paste("Regressogram => RMSE is", rmse, "when h is 0.37"))

#Nonparametric method 2: Running Mean Smoother
#RMSCalculation
i<-1
lengt <- length(e_train)
rmsC <- function(x){
  for(t in lengt){
    res_num<-c()
    res_denum <- c()
    
    if(abs((x - e_train[i]) / bin_width)<1)
      {
      res_num <- cbind(res_num, w_train[i])
      res_denum <- cbind(res_denum, 1)
      }
    else 
      {
      res_num <- cbind(res,0)
      res_denum <- cbind(res_denum, 0)
      }
    init <- init + 1
  } 
  return(sum(res_num)/sum(res_denum))
}


#predicting p
p_head <- sapply(data_interval, function(x) {
  return(rmsC(x))
})

#plotting
plot(e_train, w_train, type = "p", pch = 19, col = "blue", las = 1, main = sprintf("h = %g", bin_width))
points(x=e_test,y=w_test, col = "purple",las = 1,pch = 19)
for (b in 1:length(data_interval)) {
  lines(c(left_borders[b], data_interval[b+1]), c(p_head[b], p_head[b]), lwd = 2, col = "black")
  if (b < length(data_interval)) {
    lines(c(data_interval[b], data_interval[b+1]), c(p_head[b], p_head[b + 1]), lwd = 2, col = "black") 
  }
}

init <- 1
N_test <- length(e_test)
merge <- c()
for(t in N_test){
  window <- (e_test[init]-minimum_value)/bin_width
  merge <- cbind(merge, (w_test[init]-p_head[window])^2)
  init <- init + 1
}

#RMSE 
mse <-sum(merge)/N
rmse <- sqrt(mse)
print(paste("Running Mean Smoother => RMSE is", rmse, "when h is 0.37"))



#Nonparametric method 3: Kernel
lengt <- length(e_train)
init=1
#KernelCalculation
kC <- function(x){
  for(t in lengt){
    res_num<-c()
    res_denum <- c()
    kernel_num <- (1/sqrt(2*pi))*exp(-(x - e_train[init]) / bin_width^2/2)
    res_num <- cbind(res_num, kernel_num*w_train[init])
    kernel_denum <- (1/sqrt(2*pi))*exp(-((x - e_train[initiali]) / bin_width)^2/2)
    res_denum <- cbind(res_denum, kernel_denum)
    init <- init + 1
  } 
  return(sum(res_num)/sum(res_denum))
}

#predicting p
p_head <- sapply(data_interval, function(x) {
  return(kCN(x))
})

#plotting
plot(e_train, w_train, type = "p", pch = 19, col = "blue", las = 1, main = sprintf("h = %g", bin_width))
points(x=e_test,y=w_test, col = "purple",las = 1,pch = 19)
for (b in 1:length(data_interval)) {
  lines(c(left_borders[b], data_interval[b+1]), c(p_head[b], p_head[b]), lwd = 2, col = "black")
  if (b < length(data_interval)) {
    lines(c(data_interval[b], data_interval[b+1]), c(p_head[b], p_head[b + 1]), lwd = 2, col = "black") 
  }
}

init <- 1
N_test <- length(e_test)
merge <- c()
for(t in N_test){
  window <- (e_test[init]-minimum_value)/bin_width
  merge <- cbind(merge, (w_test[init]-p_head[window])^2)
  init <- init + 1
}

#RMSE 
mse <-sum(merge)/N
krmse <- sqrt(mse)
print(paste("Kernel Smoother => RMSE is", krmse, "when h is 0.37"))

