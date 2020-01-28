####### GET DATA ##############
data_set <- read.csv("hw05_data_set.csv")

x <- data_set$eruptions
y <- data_set$waiting

x_train <- x[1:150]
y_train <- y[1:150]

x_test <- x[151:272]
y_test <- y[151:272]
len_y_test <- length(y_test)

###### ERROR CALCULATION FUNC ##########
error_calculation <- function(y_train, index, error){
  mean <- mean(y_train[index])
  sum_squared_error <- sum((y_train[index] - mean) ^ 2)
  error <- error + sum_squared_error
  return(error)
}

rmse_calculation <- function(y_test, y_predicted){
  sum_squared <- sum((y_test - y_predicted) ^ 2)
  rmse <- sqrt(sum_squared / len_y_test)
  return(rmse)
}

dt_regression <- function(P) {
  splits <- c()
  means <- c()
  index_list <- list(1:150)
  terminal <- c(FALSE)
  branch <- c(TRUE)
  
########## FIT #########################
  
  while (1) {
    branching_nodes <- which(branch)
    len_branching_nodes <- length(branching_nodes)
     
    if (len_branching_nodes < 1) {
      break
    }
    
    for (node in branching_nodes) {
      
      index_node <- index_list[[node]]
      branch[node] <- FALSE
      node_mean <- mean(y_train[index_node])
      x_indices <- x_train[index_node]
      
      if (length(x_indices) <= P) {
        terminal[node] <- TRUE
        means[node] <- node_mean
      } else {
        terminal[node] <- FALSE
        pos <- (sort(unique(x_indices))[-1] +
                  sort(unique(x_indices))[-length(sort(unique(x_indices)))]) / 2
        scores <- rep(0, length(pos))
        
        for (i in 1:length(pos)) {
          error <- 0
          left_pos_check <- x_indices <= pos[i]
          right_pos_check <- x_indices > pos[i]
          
          left_indices <- index_node[which(left_pos_check)]
          right_indices <- index_node[which(right_pos_check)]
          
          len_left_ind <- length(left_indices)
          len_right_ind <- length(right_indices)
          len_ind <- len_left_ind + len_right_ind
          
          if (length(left_indices) > 0) {
            error <- error_calculation(y_train, left_indices, error)
          }
          
          if (length(right_indices) > 0) {
            error <- error_calculation(y_train, right_indices, error)
          }
          
          scores[i] <- error / len_ind
        }
        
        if (length(sort(unique(x_indices))) == 1) {
          terminal[node] <- TRUE
          means[node] <- node_mean
          next
        }
        
        best_split <- pos[which.min(scores)]
        splits[node] <- best_split
        
        left_split_check <- x_indices < best_split
        right_split_check <- x_indices >= best_split
        
        left_child <- 2 * node
        right_child <- (2 * node) + 1
        
        left_indices <- index_node[which(left_split_check)]
        right_indices <- index_node[which(right_split_check)]
        
        index_list[[left_child]] <- left_indices
        index_list[[right_child]] <- right_indices
        
        terminal[left_child] <- FALSE
        terminal[right_child] <- FALSE
        
        branch[left_child] <- TRUE
        branch[right_child] <- TRUE
      }
    }
  }
  return(list("splits"= splits, "means"= means, "terminal"= terminal))
}

######## MODEL ###########
P <- 25
dt <- dt_regression(P)
splits <- dt$splits
means <- dt$means
terminal <- dt$terminal

####### PREDICT ##########

predict <- function(m, splits, means, terminal){
  i <- 1
  while (1) {
    if (terminal[i] == TRUE) {
      return(means[i])
    } else if (terminal[i] != TRUE) {
      if (m <= splits[i]) {
        i <- i * 2
      } else if (m > splits[i]){
        i <- i * 2 + 1
      }
    }
  }
}

####### RMSE ############

y_predicted <- rep(0, 122)
for (i in 1:122) {
  y_predicted[i] <- predict(x_test[i], splits, means, terminal)
}
rmse <- rmse_calculation(y_test, y_predicted)

####### PLOT ############
point_colors <- c("blue", "red")
z <- c(rep(1, 150), rep(2, 122))

plot(x, y, type = "p", pch = 19, col = point_colors[z],
     ylim = c(min(y), max(y)), xlim = c(min(x), max(x)),
     ylab = "Waiting Time", xlab = "Eruptions", las = 1, main = "P = 25")

legend("topleft", legend = c("Training Points", "Test Points"),
       col = point_colors, pch = 19, cex = 0.75)

fit_left <- seq(from = 1.5, to = 5.1, length.out = 37)
fit_right <- seq(from = 1.6, to = 5.2, length.out = 37)

for (i in 1:37) {
  lines(c(fit_left[i], fit_right[i]), 
        c(predict(fit_left[i], splits, means, terminal), 
          predict(fit_left[i], splits, means, terminal)),
        col = "black")
  lines(c(fit_right[i], fit_right[i]), 
        c(predict(fit_left[i], splits, means, terminal), 
          predict(fit_right[i], splits, means, terminal)),
        col = "black")
}
######## P = 5 to P = 50 ################

x_P <- seq(from = 5, to = 50, by = 5)
rmse_for_p <- seq(from = 1, to = 10)

for (P in x_P) {
  dt <- dt_regression(P)
  splits <- dt$splits
  means <- dt$means
  terminal <- dt$terminal
  y_predicted <- rep(0, 122)
  for (i in 1:122) {
    y_predicted[i] <- predict(x_test[i], splits, means, terminal)
  }
  rmse_for_p[P] <- rmse_calculation(y_test, y_predicted)
  print(paste("RMSE is", rmse_for_p[P],"when P =", P))
}

M <- 5
y_P <- seq(from = 1, to = 10)
for(j in 1:10){
  y_P[j] <- rmse_for_p[M]
  M <- M + 5 
}

########## FINAL PRINTOUTS ############
print(paste("RMSE is", rmse,"when P = 25"))

########## RMSE PLOTS FOR P = 5 to P = 50 ##############
plot(x_P, y_P, type = "o", ylim = c(5, 8), xlim = c(0, 50),
     las = 1, pch = 16, xlab = "P", ylab = "RMSE")
