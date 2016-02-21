### Load in data, generate TF-IDF features.

source("~/Desktop/photo_kaggle/setting_wd.R") #set wd
library(data.table)
library(tidyr)

train <- fread("./data/training.csv")
test <- fread("./data/test.csv")

##----------------------------------------------------------------------------------
## TF-IDF functions
##----------------------------------------------------------------------------------

#TF-IDF: document is a "name" or "description/caption." For every word in each document,
# need to (1) find document-normalized word counts per document,
# (2) count number of documents that contain word of interest
# (3) obtain IDF: log(number of documents)

dictionary <- function(dt, cnames){
  if (length(cnames) == 1){
    return(unique(unlist(lapply(dt$name, FUN = function(x){strsplit(x, " ")}))) )
  }
  else{
    tmp <- NULL
    for(col in cnames){
      tmp <- union(tmp, unique(unlist(lapply(dt[[col]], FUN = function(x){strsplit(x, " ")}))))
    }
    return(tmp)
  }
}

# ### TEST:
# name_dictionary <- dictionary(train, "name") #1952 words
# desc_caption_dictionary <- dictionary(train, c("description", "caption")) #2149 words

term_freq <- function(word, doc){ #doc is string of words that has been tokenized
  count <- sum(as.numeric(word == doc))
  return(count/length(doc))
}

word_in_doc_df <- function(dt, cnames = "name", dict){
  tmp <- dt[, .SD, .SDcols = cnames]
  if(length(cnames) > 1){
    tmp <- tidyr::unite(tmp, new_vec, 1:length(cnames), sep = " ")
  }
  word_count_df <- as.data.frame(matrix(0, nrow = nrow(dt), ncol = length(dict)))
  colnames(word_count_df) <- dict
  
  for(i in 1:nrow(dt)){
    words <- strsplit(unlist(tmp[i]), " ")[[1]]
    words <- words[words != ""]
    for(word in words){
      word_count_df[i, word] <- 1
    }
  }
  
  if(dim(word_count_df)[2] > length(dict)) word_count_df <- word_count_df[,1:length(dict)]
  return(word_count_df)
}

# ### TEST:
# names_word_occurrence <- word_in_doc_df(train, cnames = "name", dict = name_dictionary)
# names_n_docs_containing <- colSums(names_word_occurrence)
# 
# desc_caption_word_occurrence <- word_in_doc_df(train, cnames = c("description", "caption"), dict = desc_caption_dictionary)
# desc_caption_n_docs_containing <- colSums(desc_caption_word_occurrence)


tfidf <- function(word, doc, n_doc_list){
  if(word %in% names(n_doc_list)){
    idf <- log(nrow(train)/1+(n_doc_list[[word]])) #load n_docs containing word value
    return(term_freq(word, doc) * idf)
    } else return(0)
}


#Wrapper function for obtaining tf-idf data:
text_features <- function(dt, cnames, dictionary_dt){
  dict <- dictionary(dictionary_dt, cnames) #make a dictionary of all words in cnames columns of dictionary_dt
  word_occurrence <- word_in_doc_df(dt, cnames, dict = dict) #count number of documents each word in dict occurs
  n_docs_containing <- colSums(word_occurrence) 
  
  tfidf_df <- as.data.frame(matrix(0, nrow = nrow(dt), ncol = length(dict))) #storage
  colnames(tfidf_df) <- dict
  
  if(length(cnames) > 1){
    tmp <- tidyr::unite(dt[, .SD, .SDcols = cnames], new_vec, 1:length(cnames), sep = " ") #combine text from multiple columns
  } else{
    tmp <- dt[[cnames]]
  }
  
  for(i in 1:nrow(dt)){
    if(i%%5000==0){
      print(sprintf("finished analyzing tf-idf for %d rows of data.table...", i))
    }
    
    doc <- strsplit(as.character(tmp[i]), " ")[[1]] #split character vector of integers
    
    if(length(doc)>0){
      for(word in doc){
        if(word != ""){
          tfidf_df[i, word] <- tfidf(word, doc, n_docs_containing) #get tf-idf for each word in per 'doc' in an album
        }
      }
    }
  }

  if(dim(tfidf_df)[2] > length(dict)) tfidf_df <- tfidf_df[,1:length(dict)]
  return(tfidf_df)
}

##----------------------------------------------------------------------------------
## Obtain TF-IDF features for training and test sets:
##----------------------------------------------------------------------------------

train_name_tfidf <- text_features(train, "name", train)
train_desc_caption_tfidf <- text_features(train, c("caption", "description"), train)

test_name_tfidf <- text_features(test, "name", train)
test_desc_caption_tfidf <- text_features(test, c("caption", "description"), train)

##----------------------------------------------------------------------------------
## Save TF-IDF data
##----------------------------------------------------------------------------------

saveRDS(train_name_tfidf, "./data/train_name_tfidf.rds")
saveRDS(train_desc_caption_tfidf, "./data/train_desc_caption_tfidf.rds")

saveRDS(test_name_tfidf, "./data/test_name_tfidf.rds")
saveRDS(test_desc_caption_tfidf, "./data/test_desc_caption_tfidf.rds")

