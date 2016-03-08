##----------------------------------------------------------------------------------
## GET PREDICTIONS
##----------------------------------------------------------------------------------

source("~/Desktop/photo_kaggle/setting_wd.R") #set working directory.
source("./src/dt_preprocessing.R")
library(data.table); library(xgboost); library(caret); library(pROC); library(ROCR); library(ggplot2)
set.seed(123)

##----------------------------------------------------------------------------------
## Training data preprocessing, appending/modifying features.
##----------------------------------------------------------------------------------

assemble_data <- function(train_or_test = "train",
                          data_file = "./data/training.csv",
                          name_tfidf_file = "./data/train_name_tfidf.rds",
                          desc_caption_tfidf_file = "./data/train_desc_caption_tfidf.rds",
                          country_file = "./data/aggregate_train_countries.RDS"){
  dt <- fread(data_file)
  dt <- remove_columns(add_area(dt))

  if(train_or_test == "train"){
    print("Assembling training data...")
    y <- as.matrix(as.numeric(dt[["good"]]))
    dt[, "good":=NULL]
  } else print("Assembling test data for predictions...")

  name_tfidf <- readRDS(name_tfidf_file); desc_caption_tfidf <- readRDS(desc_caption_tfidf_file)
  country <- data.frame("country" = readRDS(country_file))

  #If assembling test data, impute the most common country if country name is missing.
  if(train_or_test == "test"){
    country$country[which(is.na(country))] <- names(which.max(table(country)))
    } else{
    rm_inds <- which(is.na(country))
  }

  country_one_hot <- model.matrix(~0+country, data = country)

  df_text <- cbind(dt, name_tfidf) #names text data
  df_text <- cbind(df_text, desc_caption_tfidf) #description/caption text data
  colnames(df_text) <- make.names(colnames(df_text), unique = T)

  if(train_or_test == "train"){
    df_text <- df_text[-rm_inds, ]
    y <- y[-rm_inds]
    return(list(data = as.matrix(cbind(df_text, country_one_hot)), y = y))
  } else{
    return(list(data = as.matrix(cbind(df_text, country_one_hot))))
  }
}

train_data <- assemble_data()
train <- train_data$data; y <- train_data$y


##----------------------------------------------------------------------------------
## For an example, fit one xgboost model with random parameters.
##----------------------------------------------------------------------------------

#split data into training/test sets:
trainIndex <- as.vector(caret::createDataPartition(y = y,
                                         p = 0.8,
                                         list = FALSE))
#store explicitly the training data
train.DMat <- xgb.DMatrix(data = train[trainIndex, ], label = y[trainIndex])

#store explicitly the validation data
valid.DMat <- xgb.DMatrix(data = train[-trainIndex, ], label = y[-trainIndex])

#hyperparameters
max.depth <- 8 #How deep each weak learner (small tree) can get. Will control overfitting.
eta <- 0.5 #eta is the learning rate, which also has to do with regularization. 1 ->> no regularization. 0 < eta <= 1.
nround <- 150 #The number of passes over training data. This is the number of trees we're ensembling.

#Fit boosted model with our random parameters. Save output that would otherwise print to console.
sink("./data/watchlist_output.txt", append = FALSE)
bst <- xgb.train(data = train.DMat,
                watchlist = list(train = train.DMat, validation = valid.DMat),
                max.depth = max.depth,
                eta = eta, nthread = 4,
                nround = nround,
                objective = "binary:logistic",
                eval_metric = "logloss")
sink()

#View training/validation logloss metrics:
bst_output <- read.table("./data/watchlist_output.txt", sep = "\t")
# which.min(sapply(bst_output[,3], FUN = function(x){as.numeric(substr(x, 14, 21))}))

valid.preds <- predict(bst, valid.DMat) #Xgboost::predict returns class probabilities, not class labels!
theta <- 0.5
valid.class_preds <- ifelse(valid.preds >= theta, 1, 0)
valid.accuracy <- sum(as.numeric(valid.class_preds == y[-trainIndex]))/length(y[-trainIndex]) #~78% valid accuracy
print(sprintf("Accuracy on validation set: %f", valid.accuracy))

## Variable importance:
importance_matrix <- xgb.importance(model = bst, feature_names = colnames(train))
print(importance_matrix)
barplot(importance_matrix$Gain[6:1],
        horiz = T,
        names.arg = importance_matrix$Feature[6:1],
        main = "Estimated top 6 features by accuracy gain",
        cex.names = .6)


##----------------------------------------------------------------------------------
## Tune xgb model with a gridsearch over parameters and cross validation.
##----------------------------------------------------------------------------------

#Define hyperparameter grid. This one will call for 8 models.
#Note, the parameter names in caret are different than in xgb.train.
xgb_grid <- expand.grid(nrounds = c(100),
                        eta = c(0.2),
                        max_depth = c(7, 10),
                        colsample_bytree = c(0.6, 1),
                        gamma = c(0.75, 1),
                        min_child_weight = c(1))

tr_control <- caret::trainControl(method = "cv",
                          number = 5,
                          classProbs = TRUE, 
                          allowParallel = TRUE,
                          summaryFunction = mnLogLoss, #Use summaryFunction will use ROC (i.e. AUC) to select optimal model. Write custom one.
                          verboseIter = TRUE) 

xgb_cv1 <- caret::train(x = train,
                       y = as.factor(ifelse(y==1, "good", "bad")), #target vector should be non-numeric factors
                       tuneGrid = xgb_grid, #Which hyperparameters we'll test.
                       trControl = tr_control, #How cross validation will run.
                       method = "xgbTree",
                       metric = "logLoss",
                       maximize = FALSE)

save(xgb_cv1, file = "./output/model.RData")

## Get validation set predictions:
valid.preds <- xgboost::predict(xgb_cv1, train[-trainIndex, ], type = "prob") #Obtain class probabilities.


##----------------------------------------------------------------------------------
## Make ROC curve, get AUC statistic.
##----------------------------------------------------------------------------------

#ROC curve by hand:
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
# cutoff_value <- function(fpr, true, probs, tol = 0.0005, max_iter = 10000){
#   theta_est <- 0.5
#   fpr_tmp <- FPR(true, probs, theta_est)
#   err <- fpr_tmp - fpr
#   step <- 0.0005
#   iter <- 1
#   while((abs(err) > tol) & (iter <= max_iter)){
#     if(iter %% 500 == 0) print(sprintf("Solving for theta. On iteration %d.", iter))
#     if(sign(err) == 1){ #Too many false positives.
#       theta_est <- theta_est + step #If fpr_tmp too high, increase threshold.
#     } else{
#       theta_est <- theta_est - step
#     }
#     fpr_tmp <- FPR(true, probs, theta_est)
#     err <- fpr_tmp - fpr
#     iter <- iter+1
#   }
#   print(sprintf("Estimated fpr: %.5f", fpr_tmp))
#   return(theta_est)
# }

roc <- matrix(NA, nrow = 100, ncol = 2)
thetas <- seq(0, 1, length.out = 100)

for(i in 1:length(thetas)){
  roc[i,] <- c(FPR(true = y[-trainIndex], probs = valid.preds$good, theta = thetas[i]),
               TPR(true = y[-trainIndex], probs = valid.preds$good, theta = thetas[i]))
}

plot(roc[,1], roc[,2], type = "l", lwd = 2, col = 'blue', main = "AUC", ylab = "TPR", xlab = "FPR", xlim = c(0,1), ylim = c(0,1))
abline(a = 0, b = 1, col = 'black', lwd = 2)

# Using the pROC and ROCR packages, obtain AUC statistic and plot ROC curve:
# Note, this is adapted from a Yhat blog post: http://blog.yhat.com/posts/roc-curves.html
auc_est <- pROC::auc(y[-trainIndex], valid.preds$good) #0.8638 on the validation set, not bad...

rocr_pred <- prediction(valid.preds$good, y[-trainIndex])
rocr_perf <- performance(rocr_pred, measure = "tpr", x.measure = "fpr") #this is of class "performance," it's not a list
auc_data <- data.frame("fpr" = unlist(rocr_perf@x.values), "tpr" = unlist(rocr_perf@y.values))

#Get theta-cutoff value for test data to classify good/bad albums based on class probabilities.
# cutoff_index <- which.min(apply(auc_data, 1, FUN = function(x){norm(as.vector(c(0,1))-as.vector(x), "2")}))
# theta <- cutoff_value(auc_data$fpr[cutoff_index], y[-trainIndex], valid.preds$good)

auc_roc_curve <- ggplot(data = auc_data, aes(x = fpr, y = tpr, ymin = 0, ymax = tpr)) +
  geom_line(aes(y = tpr), alpha = 0.8, color = 'blue') + #draw ROC curve
  geom_ribbon(alpha = 0.2) + #shade under blue curve
  ggtitle(sprintf("ROC Curve illustrating AUC = %.4f", auc_est))

auc_roc_curve


##-------------------------------------------------------------------------------------------
## Process test data, get predictions
##-------------------------------------------------------------------------------------------

test_data <- assemble_data(train_or_test = "test",
                          data_file = "./data/test.csv",
                          name_tfidf_file = "./data/test_name_tfidf.rds",
                          desc_caption_tfidf_file = "./data/test_desc_caption_tfidf.rds",
                          country_file = "./data/aggregate_test_countries.RDS")

test <- test_data$data
test_classes <- ifelse(predict(xgb_cv1, test, type = "raw") == "good", 1, 0)

#Save and then submit predictions!
write.csv(data.frame(id = read.csv("./data/test.csv")$id, good = test_classes),
          "./output/test_predictions.csv", row.names = FALSE)

