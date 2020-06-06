#Source
source("DataProcessing-Lib.R")

#Packages
library(xlsx)
library(data.table)
library(keras)

#Global variables
windowSize <- 7
splitPerc <- .85

#Import data
dt <- data.table(read.xlsx("Data/dtAnalysis_New.xlsx", sheetName = "Sheet1"))

#Train-test split
split <- round(splitPerc*nrow(dt))
dtTrain <- dt[1:split]
dtTest <- dt[(split+1):nrow(dt)]

#Data table for result/error visualization
dtTrainRes <- dtTrain[(windowSize+1):nrow(dtTrain), c("Date", "NumberOfTickets")]
dtTestRes <- dtTest[(windowSize+1):nrow(dtTest), c("Date", "NumberOfTickets")]

#Set date to null
dtTrain[, Date := NULL]
dtTest[, Date := NULL]

#Data normalization
norm_scale <- fnMinMaxScale(dtTrain)
dtTrainNorm <- fnNormalizeData(dtTrain,norm_scale)
dtTestNorm <- fnNormalizeData(dtTest,norm_scale)

#Compute train and test tensors
Train <- fnComputeTensors(dtTrainNorm,windowSize)
Test <- fnComputeTensors(dtTestNorm,windowSize)

#Model definition and training
ipShape <- c(windowSize, dim(Test$inputTensor)[3])
model <- fnLstmModel(ipShape)
models <- fnLstmFit(model,Train,Test,200)

#In-sample prediction
pred <- fnLstmPredict(models$model,Train$inputTensor)

#Data denormalization
pred <- fnDenormalizeData(pred,norm_scale)

#Error computation
dtTrainRes[, Prediction := pred]
dtTrainRes[Prediction < 0, Prediction := 0]
rmse_train <- sqrt(mean((dtTrainRes$NumberOfTickets - dtTrainRes$Prediction)^2))

#Out-of-sample prediction
pred <- fnLstmPredict(models$model,Test$inputTensor)

#Data denormalization
pred <- fnDenormalizeData(pred,norm_scale)

#Error computation
dtTestRes[, Prediction := pred]
dtTestRes[Prediction < 0, Prediction := 0]
rmse_test <- sqrt(mean((dtTestRes$NumberOfTickets - dtTestRes$Prediction)^2))

#Save model
save_model_hdf5(models$model, "Models/model-newexp2.h5")

#Export results to the excel
write.xlsx(dtTrainRes, file="C:/Users/vgupta/Documents/Temp Projects/Hackathon-C06/Results/NewExp2.xlsx", sheetName="Train", row.names = FALSE)
write.xlsx(dtTestRes, file="C:/Users/vgupta/Documents/Temp Projects/Hackathon-C06/Results/NewExp2.xlsx", sheetName="Test", row.names = FALSE, append = TRUE)
