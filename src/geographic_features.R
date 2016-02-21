### Get country names from geonames API
### Make an account at http://www.geonames.org/login
source("~/Desktop/photo_kaggle/setting_wd.R")
library(data.table)
library(geonames)

##----------------------------------------------------------------------------------
## Ping Geonames API successively to gather all country names
##----------------------------------------------------------------------------------

## You'll need to make a geonames account and set the username to get query results...
## I'd make 2 accounts so you can avoid the 30k requests in a day limit.
# options(geonamesUsername="myusername1")
# options(geonamesUsername="myusername2")

options(geonamesUsername = "myusername1")

train <- fread("data/training.csv")
test <- fread("data/test.csv")

lats <- train$latitude
lons <- train$longitude

get_country <- function(lat, lon){
  country <- tryCatch({
                      results <- geonames::GNcountryCode(lat = lat, lng = lon, lang = "English", radius = 50)
                      countryName <- results$countryName
                      countryName
                      }, error = function(e){
                        NA
                      })
  country 
}

#Save country names for training data, abiding by geonames API user limits.
countries <- vector(mode = "character", length = length(lats))
N <- length(countries);
print("Beginning country acquisition from training data...")
for(i in 38001:N){
  
  countries[i] <- get_country(lats[i], lons[i])
  
  if(i %% 500 == 0){
    print(sprintf("On training album %d.", i))
    print("Previous 5 countries:")
    print(countries[(i-5):i])
    
    if(i == 30000){
        print("i = 30000, changing usernames.")
        options(geonamesUsername = "myusername2")
    }
  }
  
  if((i %% 2000 == 0)| (i == N)){
      if(i %% 2000 == 0){
          print("Saving previous 2000 training set countries.")
          saveRDS(countries[(i-1999):i], paste0("./data/countries_train_", i-1999, "_", i, ".RDS"))
          print("System sleeping for 3650 seconds...")
          Sys.sleep(3650)}
      if(i == N){
          print("Saving the last bit of training set countries")
          leftover <- N%%2000
          saveRDS(countries[(i-(leftover-1)):i], paste0("./data/countries_train_", i-(leftover-1), "_", i, ".RDS"))
      }
  }
}

#Save country names for test data, abiding by geonames API user limits.
lats_test <- test$latitude; lons_test <- test$longitude
countries_test <- vector(mode = "character", length = length(lats_test))
N_test <- length(countries_test);
print("Beginning country acquisition from test data...")
for(i_test in 1:N_test){
  
  countries_test[i_test] <- get_country(lats_test[i_test], lons_test[i_test])
  
  if(i_test %% 500 == 0){
    print(sprintf("On test album %d.", i_test))
    print("Previous 5 countries:")
    print(countries_test[(i_test-5):i_test])
    
    if((i_test + N) %% 30000 == 0){
        print("Hit 30000 requests, changing usernames.")
        options(geonamesUsername = "myusername1")
    }
  }
  
  if((i_test %% 2000 == 0)| (i_test == N_test)){
      if(i_test %% 2000 == 0){
          print("Saving previous 2000 countries...")
          saveRDS(countries_test[(i_test-1999):i_test], paste0("./data/countries_test_", i_test-1999, "_", i_test, ".RDS"))
          print("System sleeping for 3650 seconds...")
          Sys.sleep(3650)}
      if(i_test == N_test){
          print("Saving the last bit of test set countries...")
          leftover <- N_test%%2000
          saveRDS(countries_test[(i_test-(leftover-1)):i_test], paste0("./data/countries_test_", i_test-(leftover-1), "_", i_test, ".RDS"))
      }
  }
}


##----------------------------------------------------------------------------------
## Aggregate separate country names files by training/test sets
##----------------------------------------------------------------------------------

#Concatenate countries by train/test set:
aggregate_countries <- function(file_list, N, filepath){
  inds <- seq(1, N, by = 2000)
  agg_countries <- vector(mode = "character", length = N)
  for(i in inds){
    if (i < inds[length(inds)]){
      c_file <- paste0(filepath, "_", i, "_", i+1999, ".RDS")
      print(sprintf("File being read in: %s", c_file))
      agg_countries[i:(i+1999)] <- readRDS(c_file)
      } else{
      c_file <- paste0(filepath, "_", i, "_", N, ".RDS")
      print(sprintf("File being read in: %s", c_file))
      agg_countries[i:N] <- readRDS(c_file)
      }
  }
  return(agg_countries)
}

train_country_files <- list.files("./data/")[grepl("countries_train", list.files("./data/"))]
test_country_files <- list.files("./data/")[grepl("countries_test", list.files("./data/"))]
N_train <- nrow(train); N_test <- nrow(test)
filepath_train <- "./data/countries_train"; filepath_test <- "./data/countries_test"

train_countries <- aggregate_countries(train_country_files, N_train, filepath_train)
test_countries <- aggregate_countries(test_country_files, N_test, filepath_test)

saveRDS(train_countries, "./data/aggregate_train_countries.RDS")
saveRDS(test_countries, "./data/aggregate_test_countries.RDS")






