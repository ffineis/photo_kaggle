##----------------------------------------------------------------------------------
## GET PREDICTIONS
##----------------------------------------------------------------------------------

source("~/Desktop/photo_kaggle/setting_wd.R") #set wd
source("./src/dt_preprocessing.R")
library(data.table); library(xgboost); library(caret); library(pROC)


##----------------------------------------------------------------------------------
## Data preprocessing, appending/modifying features
##----------------------------------------------------------------------------------

train <- fread("./data/training.csv")
train_clean <- remove_columns(add_area(train))

y <- as.matrix(as.numeric(train_clean$good)) #previously 'good' column was character
train_clean$good <- NULL

train_name_tfidf <- readRDS("./data/train_name_tfidf.rds")
train_desc_caption_tfidf <- readRDS("./data/train_desc_caption_tfidf.rds")

#Note: should turn training_text dt into an dgCMatrix at some point - train <- xgb.DMatrix(data = train$data, label = train$label)
train_text <- as.data.table(cbind(train_clean, train_name_tfidf)) #names text data
train_text <- as.data.table(cbind(train_text, train_desc_caption_tfidf)) #description/caption text data

train_text$country <- readRDS("./data/aggregate_train_countries.RDS")


##----------------------------------------------------------------------------------
## Fit, tune XGBoost model
##----------------------------------------------------------------------------------

#split data into training/test sets:
trainIndex <- caret::createDataPartition(y = y,
                                         p = 0.8,
                                         list = FALSE)
#store explicitly the training data
train <- xgb.DMatrix(data = as.matrix(train_text[trainIndex]), label = y[trainIndex])

#store explicitly the validation data
valid <- xgb.DMatrix(data = as.matrix(train_text[-trainIndex]), label = y[-trainIndex])

#hyper-parameters
max.depth <- 10
eta <- 0.5 #learning rate: has to do with regularization. 1 ->> no regularization. 0 < eta <= 1.
nthread <- 4
nround <- 100 #number of passes over training data. This is the number of trees we're ensembling.
             # If nrounds too big, could lead to over-fitting. Is tuning this referred to as "early stopping"?

#fit boosted model, one way:
bst <- xgboost(data = as.matrix(train_text[trainIndex]), label = y[trainIndex],
               max.depth = max.depth,
               eta = eta, nthread = nthread,
               nround = nround,
               objective = "binary:logistic",
               eval_metric = "logloss")

valid_preds <- predict(bst, as.matrix(train_text[-trainIndex])) #class probabilities, not class labels!
theta <- 0.5
valid_preds_binary <- ifelse(valid_preds >= theta, 1, 0)
valid_accuracy <- sum(as.numeric(valid_preds_binary == y[-trainIndex]))/length(y[-trainIndex]) #~76% valid accuracy
print(sprintf("Accuracy on validation set: %f", valid_accuracy))


#fit boosted model, another way: use watchlist to note where we should do early stopping -
# Here, we will watch the progression: validation logloss starts increasing due to overfitting, saving
# the text output to find the nround value corresponding to lowest test log loss.
nround <- 100

f <- file("./data/watchlist_output.txt", open = "wt")
sink(f)
bst_watchlist <- xgb.train(data = train, 
                 watchlist = list(train = train, test = valid),
                 max.depth = max.depth,
                 eta = eta, nthread = nthread,
                 objective = "binary:logistic",
                 eval_metric = "logloss")
sink()

bst_output <- read.table("./data/watchlist_output.txt", sep = "\t")
nround <- which.min(sapply(bst_output[,3], FUN = function(x){as.numeric(substr(x, 14, 21))}))


### Variable importance ###
importance_matrix <- xgb.importance(model = bst)
print(importance_matrix)
xgb.plot.importance(importance_matrix)


### Cross validation with xgb.cv ###
#Documentation: https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
parameters = list("objective" = "binary:logistic", #probability, not logits. Get logits with binary:logitraw.
                  "eval_metric" = "logloss")
cv.folds <- 10
total_train <- as.matrix(train_text)

etas = c(0.01, 0.1, 0.5, 0.75, 1); eta_stor <- list(); bst.cv <- list()
for(j in 1:length(etas)){
  bst.cv[[j]] <- xgb.cv(param = parameters,
                   data = total_train,
                   label = y,
                   nfold = cv.folds,
                   nthread = nthread,
                   nrounds = nround,
                   eta = etas[j],
                   early.stop.round = 5)
}

#Find a reasonable eta value
bst.cv.ind <- which.min(unlist(lapply(bst.cv, FUN = function(x){min(x$test.logloss.mean)})))
eta <- etas[bst.cv.ind]

bst.final <- xgboost(data = total_train,
                      label = y,
                      max.depth = max.depth,
                      eta = eta,
                      nrounds = nround,
                      nthread = nthread,
                      objective = "binary:logistic",
                      eval_metric = "logloss")


##----------------------------------------------------------------------------------
## Make AUC curve
##----------------------------------------------------------------------------------

FPR <- function(true, probs, theta){
  preds <- ifelse(probs >= theta, 1, 0)
  fpr <- (length(which((preds == 1) & (true == 0))))/length(true) #preds indicate positive but true is actually negative
  return(fpr)
}
TPR <- function(true, probs, theta){
  preds <- ifelse(probs >= theta, 1, 0)
  fnr <- length(which((preds == 0) & true == 1))/length(true) #false negatives: preds are negative but true is positive
  return(1-fnr)
}

roc <- matrix(NA, nrow = 100, ncol = 2)
thetas <- seq(0, 1, length.out = 100)
fitted <- predict(bst.final, total_train)
for(i in 1:length(thetas)){
  roc[i,] <- c(FPR(true = y, probs = fitted, theta = thetas[i]),
               TPR(true = y, probs = fitted, theta = thetas[i]))
}

plot(roc[,1], roc[,2], type = "l", lwd = 2, col = 'blue', main = "AUC", ylab = "TPR", xlab = "FPR", xlim = c(0,1), ylim = c(0,1))
abline(a = 0, b = 1, col = 'black', lwd = 2)

pROC::auc(y, fitted) #0.932, not bad...


##-------------------------------------------------------------------------------------------
## Process test data, get predictions
##-------------------------------------------------------------------------------------------

test <- fread("./data/test.csv")
test_clean <- remove_columns(add_area(test))

test_name_tfidf <- readRDS("./data/test_name_tfidf.rds")
test_desc_caption_tfidf <- readRDS("./data/test_desc_caption_tfidf.rds")

test_text <- as.data.table(cbind(test_clean, test_name_tfidf)) #names text data
train_text <- as.data.table(cbind(train_text, test_desc_caption_tfidf)) #description/caption text data

test_text$country <- readRDS("./data/aggregate_test_countries.RDS")

