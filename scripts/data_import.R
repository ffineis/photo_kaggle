### Import photo-album classification data:
source("~/Desktop/photo_kaggle/setting_wd.R")
library(data.table)
library(e1071)

training_data <- fread("./data/training.csv")
training_data[, good := as.numeric(get("good"))]

one_data <- training_data[good == 1]
zero_data <- training_data[good == 0]

names_vocab <- unlist(lapply(training_data$name, FUN = function(x){strsplit(x, " ")[[1]]}))

description_vocab <- unlist(lapply(training_data$description, FUN = function(x){strsplit(x, " ")[[1]]}))

caption_vocab <- unlist(lapply(training_data$caption, FUN = function(x){strsplit(x, " ")[[1]]}))


#ideas for text data: produce probabilities from a Niave Bayes Classifier model or perceptron model,
#                     use as features to RF model

get_occur_matrix <- function(char_vec, vocab){
  
  occur_df <- as.data.frame(matrix(rep(0, length(char_vec)*length(unique(vocab))),
                         nrow = length(char_vec), ncol = length(unique(vocab))))
  colnames(occur_df) <- unique(vocab)
  
  for(i in 1:nrow(occur_df)){
    words <- strsplit(char_vec[i], " ")[[1]]
    for(w in words){
      occur_df[i, w] <- 1
    }
  }
  return(occur_df)
}

names_occur <- get_occur_matrix(training_data$name, names_vocab)
description_occur <- get_occur_matrix(training_data$description, description_vocab)
caption_occur <- get_occur_matrix(training_data$name, caption_vocab)

nb = naiveBayes(training_data$name ~ ., data=names_occur)
# predict(nb, new_data, type = c("raw")) #get class probabilities

usv_names <- svd(names_occur)
components_names <- usv$v[,2:3]
projected_data <- as.matrix(names_occur)%*%components_names
plot(projected_data[,1], projected_data[,2], pch = 16, col = ifelse(training_data$good==0, "red", "blue"), cex = .6)


#feature mining: there are 8 words in names with counts over 1000
