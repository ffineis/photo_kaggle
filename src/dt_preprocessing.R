##----------------------------------------------------------------------------------
## Functions for data preprocessing
##----------------------------------------------------------------------------------

### Helper functions for preprocessing of training/test data:
### We'll need to remove id column, name/caption/description columns, and potentially standardize columns...

add_area <- function(dt){
  if(all(c("width", "height") %in% names(dt))){
    dt[, "area":=get(eval('width'))*get(eval('height'))]
    return(dt)
  }
}

remove_columns <- function(dt, cnames = c("name", "description", "caption", "id")){
  cnames <- cnames[cnames %in% names(dt)]
  print("removing columns:")
  print(cnames)
  return(dt[, .SD, .SDcols = names(dt)[!(names(dt) %in% cnames)]])
}

standardize_columns <- function(dt, cnames = c("width", "height", "area", "size")){
  dtc <- copy(dt)
  tmp <- dtc[, .SD, .SDcols = cnames]
  std_devs <- apply(tmp, 2, sd)
  means <- colMeans(tmp)
  
  tmp_std <- sweep(sweep(tmp, 2, means, "-"), 2, std_devs, "/")
  for (col in cnames) set(dtc, j = col, value = tmp_std[[col]])
  
  return(dtc)
}


# ### TEST:
# library(data.table)
# source("~/Desktop/photo_kaggle/setting_wd.R") #set wd
# train <- fread("./data/training.csv")
# train_std <- standardize_columns(remove_columns(add_area(train)))