# Food-amenities-demand-prediction
Predicting the demand of food amenities using LSTM and 3-layer neural network.

***Business Problem***
* Predicting the demand qunatity of food amenities
* No input is at disposal, hence the input variables need to be forecasted and then the target variable is regressed through the forecasted input variable
* Stock Keeping Units (SKUs) under consideration - Cucumber (Indian), Carrot (local), Ridge Gourd 

**Data Definition**
> Data Variables and Definition
* Input variables
1. AvgSP - Average Selling Price of SKU
2. OP - Average Selling Price of Onion
3. CustomerCount - Total GT Customers for the given SKU ( = CustomerCount + Missed Customers)
* Target Variable - ActualDemand of SKU ( = Ordered Quantity + Missed Demand)

> Time Period considered - 17/03/2017 to 22/06/2017

**Data Understanding and Processing**
> Outlier Treatment
* Values below 3rd percentile of the sample and above 97th percentile of the sample are converted to their respective buffers
* Only @CustomerCount and @ActualDemand are considered for outlier treatment

> Summary Statistics
* Summary Stats for Cucumber (Indian)

> Training and Test Datasets
* The last week of the complete dataset is considered for testing while the rest of the dataset is considered for training

> Function to create Data Input to model
* AvgSP
1. @AvgSP is predicted using time series forecasting.
2. Long Short-Term Memory (Recurrent Neural Network) method is used for forecasting. The forecasting problem is now considered as a supervised learning problem where the input is the value prior to the target day.
3. LSTM is a special type of Neural Network which remembers information across long sequences to facilitate the forecasting.
4. Forecasting results
a. Cucumber (Indian) - 
b. Carrot (local)
c. Ridge Gourd - 

* CustomerCount
1. @CustomerCount is predicted using the same method as @AvgSP
2. Forecasting Results
a. Cucumber (Indian) - 
b. Carrot (local) - 
c. Ridge Gourd - 

* Onion Price is known with good accuracy due to information about the lot size.

**Data Modelling**
> Model Name
* 3-layer Neural Network using Keras Library (tensorflow backend)
* The network is made up of 3 layers:

1. Input layer
- Takes input variables and converts them into input equation
- Parameters: no. of neurons (memory blocks) = 16, activation function = linear, weight initializer = normal distribution, kernel and activity regularizer = L1 (alpha = 0.1)

2. Hidden Layer
- The processing (optimization) takes place in this layer.
- Parameters: no. of neurons = 8, activation function = linear, weight initializer = normal distribution, kernel and activity regularizer = L1 (alpha = 0.1)

3. Output Layer
- Converts the processed results into a reverse scaled output.

> Model Performance
1. Cucumber (Indian)

2. Carrot (local)

3. Ridge Gourd
