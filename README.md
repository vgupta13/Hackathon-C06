# Hackathon-C06
An LSTM-based solution to predict the number of IT-tickets for the next day.

## Problem Formulation
The number of IT-Tickets prediction problem is formulated as supervised sequence learning task. The next day value is predicted using the N-consecutive previous days data.

## Analysis Data Preparation
The data from "Störungen" and "Anforderungen" sheets is joined and aggregated to obtain the number of IT-Ticktes value per day. Also, the date and holiday features (including lags and leads) are engineered and incorporated in the label encoding scheme.

## Data Preprocessing
The data is embedded to lower euclidean dimensions, a.k.a. time series sliding window. The sliding window size is set to 7 (and 14). The data is then split to train and test.
The train data is normalized to [0,1] and the same scale (of training data) is applied over the test data.

## Model Architecture
A 2-layer LSTM model is created with 32 cells in each layer. To learn more about the hyperparameters setting, please check DataProcessing-Lib.R file.

## Results
Please check Results and Experiments folders for the result
