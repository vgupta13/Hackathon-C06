#Data preprocessing functions 

#Min-Max scale
fnMinMaxScale <- function(dtTrain){
  norm_scale <- list()
  norm_scale$mins  <- apply(dtTrain, 2, min)
  norm_scale$maxs  <- apply(dtTrain, 2, max)
  return(norm_scale)
}

#Data normalization
fnNormalizeData <- function(dt,norm_scale){
  dt <- data.table(scale(dt, center = norm_scale$mins, 
                             scale = norm_scale$maxs - norm_scale$mins))
  return(dt)
}

#Time series data preparation
fnComputeTensors <- function(dt,windowSize){
  inlen <- ncol(dt)
  
  #Output tensor
  outputTensor <- dt[(windowSize+1):nrow(dt),][["NumberOfTickets"]]
  
  
  #Input tensor
  #Time series creation
  dt <- embed(as.matrix(dt), dimension = windowSize)
  
  #Loop to reverse the sequence 
  #i.e.from t,t-1,t-2...t-windowSize to t-windowSize...t-2,t-1,t
  for (i in windowSize:1) {
    startCol <- inlen*(i-1) + 1
    endCol   <- inlen*i
    if (i == windowSize) {
      inputTensor_temp <- dt[,startCol:endCol]
    } else {
      inputTensor_temp <- cbind(inputTensor_temp,dt[,startCol:endCol])
    }
  }
  
  #Remove rows from bottom
  truncRow <- c(nrow(inputTensor_temp):nrow(inputTensor_temp))
  inputTensor <- inputTensor_temp[-truncRow,]
  
  #Reshape input into 3D shape [samples, timesteps, features] 
  inputTensor <- array(inputTensor, c(nrow(inputTensor), windowSize, ncol(inputTensor)/windowSize))
  lst <- list()
  lst$inputTensor <- inputTensor
  lst$outputTensor <- outputTensor
  return(lst)
}

# Function to design the LSTM Network
fnLstmModel <- function(ipShape){
  # Design the LSTM network
  model <- keras_model_sequential()
  model %>%
    layer_lstm(units = 64, input_shape = ipShape, return_sequences = TRUE) %>%
    layer_lstm(units = 64, return_sequences = FALSE) %>%
    layer_dense(units = 1)
  model %>% compile(
    loss = 'mae', 
    optimizer = 'adam',
    metrics = c('accuracy')
  )
  
  return(model)
}

#Fit data to the LSTM
fnLstmFit <- function(model,Train,Test,epochs) { 
  Train_X <- Train$inputTensor
  Train_Y <- Train$outputTensor
  Test_X <- Test$inputTensor
  Test_Y <- Test$outputTensor
  
  # Fit the model
  history <- model %>% fit(
    Train_X, Train_Y, verbose = 1,
    epochs = epochs, batch_size = 8,
    validation_data = list(Test_X, Test_Y), 
    shuffle = TRUE
  )
  models <- list()
  models$model <- model
  models$history <- history
  return(models)
}

#Function to make a prediction
fnLstmPredict <- function(model, inputData){
  predict <- model%>%predict(inputData)
  return(predict)
}

#Function to denormalize the data
fnDenormalizeData <- function(data,norm_scale){
  min  <- norm_scale$mins["NumberOfTickets"]
  max  <- norm_scale$maxs["NumberOfTickets"]
  diff <- max-min
  data <- round(data*diff + min)
  return(data)
}
