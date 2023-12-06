data <- read.csv("../../data/data_cleaned.csv")

#replace "" values with missing
data[data == ""] <- NA

categorical_columns <- sapply(data, function(x) is.character(x))
data[categorical_columns] <- lapply(data[categorical_columns], as.factor)

library(missRanger)
data_cleaned_imputed <- missRanger(data, num.trees = 100, verbose = 1)

write.csv(data_cleaned_imputed, file = "../../data/data_imputed_R3.csv", row.names = FALSE)

#remove state column
data <- data[,-1]

#register parallel backend
library(doParallel)
cl <- makeCluster(4)
registerDoParallel(cl)

#missForest
library(missForest)
data_test <- missForest(data, maxiter = 10, ntree = 100, verbose = TRUE, 
                           parallelize = "forests")


library(missRanger)
data_test <- missRanger(data, num.trees = 100, verbose = 1)

#add state column back with column name state

data1 <- cbind(data$state, data_imputed)
colnames(data1)[1] <- "state"




#export to csv without index
write.csv(data_test, file = "../../data/data_imputed_R3.csv", row.names = FALSE)

#get imputed rows in data_imputed
which(is.na(data$chd))
#print row 210 of data
row = data[210,]
row_imputed = data_imputed[210,]

#missing rows in data
missing_index = which(is.na(data$chd))

#rows of data_imputed with missing_index
missing_rows = data[missing_index,]
missing_rows_imputed = data_imputed[missing_index,]

#add state column to data_imputed

colSums(is.na(data_cleaned_imputed))
#unique values in chd
unique(data$chd)


