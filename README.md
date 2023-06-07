# House-Price-Prediction

## Abstract:- 

In this study, the machine learning algorithms k-Nearest-Neighbours regression (k-NN) and Random Forest (RF) regression were used to predict house prices from a set of features in the Ames housing data set. The algorithms were selected from an assessment of previous research and the intent was to compare their relative performance at this task. Software implementations for the experiment were selected from the scikitlearn Python library and executed to calculate the error between the actual and predicted sales price using four different metrics. Hyperparameters for the algorithms used were optimally selected and the cleaned data set was split using five-fold cross-validation to reduce the risk of bias. An optimal subset of hyperparameters for the two algorithms was selected through the grid search algorithm for the best prediction. The Random Forest was found to consistently perform better than the kNN algorithm in terms of smaller errors and be better suited as a prediction model for the house price problem. With a mean absolute error of about 9 % from the mean price in the best case, the practical usefulness of the prediction is rather limited to making basic valuations.

## Data Collection:-

Data collection methods can vary depending on the type of data and the research question. Some common methods of data collection include:
1.	Surveys and questionnaires: These are structured forms that are used to gather    information from a group of people. Surveys and questionnaires can be conducted online, by mail, or in person.
2.	Interviews: Interviews involve asking questions to an individual or group of people to obtain qualitative data.
3.	Observation: This involves observing people or events to gather data. Observations can be conducted in person or through video recordings.
4.	Secondary sources: Secondary sources include data that has already been collected and is available in published reports, articles, or other sources.
5.	Sensor data: Sensor data is collected using sensors or devices that can capture information about physical properties such as temperature, humidity, or location.

## Data pre-processing:-

Data preprocessing is a crucial step in any data analysis or machine learning project. It involves transforming raw data into a format that can be used for analysis or modeling. Data cleaning is one of the important steps in data preprocessing, where missing or irrelevant data is removed from the dataset.
Missing data can occur due to various reasons such as data entry errors, data corruption, or incomplete data. Missing data can affect the accuracy of the analysis or model, and it is important to handle it appropriately. One approach is to remove any records that have missing data. Another approach is to impute the missing values using statistical methods such as mean, median, or mode imputation.
Irrelevant data refers to the data that is not useful or does not contribute to the analysis or model. This type of data can also be removed from the dataset. For example, in the house price forecasting project, if a feature such as 'url' does not provide any useful information for the analysis or model, it can be removed.
Other preprocessing steps may include scaling or normalizing the data, transforming the data into a different format or data type, or encoding categorical variables.
Overall, data preprocessing is essential to ensure that the data is clean, consistent, and ready for analysis or modeling.
 



## Exploratory Data Analysis:-

(EDA) is an important step in any data analysis or machine learning project. It involves analyzing and visualizing the data to gain insights and identify patterns, trends, or relationships between the features and the target variable.
In the house price forecasting project, EDA can help to understand the distribution and relationships of the features and target variable. For example, we can visualize the distribution of the target variable (house prices) using histograms, box plots, or density plots. We can also examine the relationships between the target variable and the numerical features using scatter plots or correlation matrices.
EDA can also help to identify any outliers or anomalies in the data. Outliers are data points that are significantly different from other data points and can have a significant impact on the analysis or model. Identifying and handling outliers appropriately is important for accurate analysis or modeling.
Another important aspect of EDA is examining the distribution and relationships of categorical variables. For example, we can use bar plots or stacked bar plots to visualize the frequency or proportion of different categories in categorical variables such as building type or renovation condition.

 


Overall, EDA is an important step in understanding the data and identifying any potential issues or opportunities for further analysis or modeling. It can also help into the underlying relationships or patterns in the data.to inform feature engineering or selection decisions and can provide insights. Feature engineering is the process of creating new features from the existing ones in the dataset to improve the accuracy and performance of the machine learning model. The goal of feature engineering is to extract more information from the available data, and to represent the data in a more meaningful way for the machine learning algorithm.
In the house price forecasting project, feature engineering may involve creating new features such as:
6.	Age of the property: This can be calculated by subtracting the construction year from the current year. The age of the property may have an impact on the house price, as older properties may require more maintenance or may not have modern amenities.

 

7.	Price per square meter: This can be calculated by dividing the total price by the square footage of the house. This feature can be useful for comparing the prices of houses with different sizes.
8.	Total number of rooms: This can be calculated by adding the number of living rooms, drawing rooms, and bedrooms. This feature may be useful for predicting the house price, as larger houses with more rooms may be more expensive.
9.	Location-based features: These may include variables such as distance from the nearest school, park, or public transportation. These features may have an impact on the house price, as houses in more desirable locations may be more expensive.
10.	Interaction terms: These are created by multiplying two or more features together. For example, multiplying the number of bathrooms by the number of bedrooms may capture a relationship between these two variables that affects the house price.
Overall, feature engineering is an important step in improving the performance of machine learning models. It requires domain knowledge and creativity to identify and create relevant features that capture the underlying patterns and relationships in the data.

 



Model evaluation is the process of assessing the performance of the machine learning model in predicting the house prices. This is typically done using various metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).
The MAE represents the average absolute difference between the predicted and actual values of the house prices, while the MSE measures the average squared difference between the predicted and actual values. The RMSE is the square root of the MSE and provides a more interpretable metric that is in the same units as the target variable (in this case, the house price).

 

In addition to these metrics, other evaluation methods such as cross-validation and train-test splits may also be used to evaluate the model's performance. These methods help to ensure that the model is not overfitting to the training data and can generalize well to new, unseen data.
Once the model has been evaluated, its hyperparameters can be tuned to further improve its accuracy. Hyperparameters are parameters that are set before training the model and cannot be learned from the data. Examples of hyperparameters include the learning rate, regularization strength, and number of hidden layers in a neural network.
Hyperparameter tuning involves selecting the optimal values for these parameters that result in the best performance of the model on the validation data. This can be done using methods such as grid search, random search, or Bayesian optimization.
Overall, model evaluation and tuning are crucial steps in developing an accurate machine learning model for house price forecasting. They help to ensure that the model is performing well and can generalize to new data.
 

Simulation
Spearman Correlation Coefficient Analysis According to the reference research and our previous discussion in, an analysis for correlation coefficients of different variables with the housing price is conducted. With the data processing methodology using Python, Spearman correlation coefficients simulated as shown in using the housing price data set 
 
As the methodology discussed above indicates, an empirical analysis based on the Boston housing price data set is conducted to test multiple factors and their impact on the median housing price as a response variable.
In the first place, data analysis is conducted on housing price, in the sense that the influence of the number of rooms on the overall housing price is analyzed, which can be seen in the horizontal axis represents the average number of rooms per house, while the vertical axis represents the median price of self-owned houses in that region, measured in 1,000 US dollars. There exists a positive, upward-sloping relationship between the number of rooms and overall housing price. With more rooms, the house is more likely to be a superior residence with higher quality and market value. This empirical evidence and trend have been a consistency of our common sense, which indicates a property will generally sell at higher prices when it has more rooms and space for living purposes. This empirical result has cross verified the features and trends.

 

 

Conclusion
The author constructs a fundamental algorithm based on the multiple linear regression method to predict housing prices and combines it with the Spearman correlation coefficient to determine the influential factors affecting housing prices. To train and test the parameters of this multiple linear regression model, the author applies the data set of the housing prices in Boston for model construction. From the simulation results shown above, it can be concluded that the proposed multiple linear regression model can effectively analyze and predict the housing price to some extent. Admittedly, the prediction accuracy is still limited at specific points, and the universality of the model still needs to be improved in further research. In further research into the corresponding models, the author will further study machine learning in the application of housing price prediction, as well as constructing a more robust algorithm based on a more advanced machine learning methodology.

