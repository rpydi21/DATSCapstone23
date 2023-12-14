data <- read.csv("../../data/data_cleaned.csv")

#replace "" values with missing
data[data == ""] <- NA

categorical_columns <- sapply(data, function(x) is.character(x))
data[categorical_columns] <- lapply(data[categorical_columns], as.factor)

library(missRanger)
#Use missRanger Package to impute missing values
data_cleaned_imputed <- missRanger(data, num.trees = 100, verbose = 1)

#export to CSV
write.csv(data_cleaned_imputed, file = "../../data/data_imputed_R.csv", row.names = FALSE)
