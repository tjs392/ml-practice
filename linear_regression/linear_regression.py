import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv("insurance.csv")

# going to convert the non numeric columns to 1's and zeroes
data["sex"] = (data["sex"] == "male").astype(int)
data["smoker"] = (data["smoker"] == "yes").astype(int)

# here we need to one hot encode regions into their separate region
# so we can habe good data
data = pd.get_dummies(data, columns=["region"], drop_first=True)

# now shuffle the data
np.random.seed(0)
shuffled_data = data.sample(frac=1, random_state=(0)).reset_index(drop=True)

# separate into features and targets, because we will be using the features to predict the charges
features = shuffled_data.drop("charges", axis=1)
target = shuffled_data["charges"]

# the split the data into 2/3 training and 1/3 testing
split = int(len(shuffled_data) * (2/3))
features_training = features[:split]
features_testing = features[split:]
target_training = target[:split]
target_testing = target[split:]

# train the model and predict w/ simple lin regressoin
linear_regression = LinearRegression(fit_intercept=True)
linear_regression.fit(features_training, target_training)

target_training_prediction = linear_regression.predict(features_training)
target_testing_predicition = linear_regression.predict(features_testing)

def smape(y, y_hat):
    numerator = np.abs(y - y_hat)
    denominator = np.abs(y) + np.abs(y_hat)
    return (1/len(y)) * np.sum(numerator / denominator)

# calc the errors
training_smape = smape(target_training, target_training_prediction)
testing_smape = smape(target_testing, target_testing_predicition)

training_rmse = np.sqrt(mean_squared_error(target_training, target_training_prediction))
testing_rmse = np.sqrt(mean_squared_error(target_testing, target_testing_predicition))
   
print(f"Calculated Bias:            {linear_regression.intercept_}")
print(f"RMSE of Training Data:      {training_rmse:.2f}")
print(f"RMSE of Testing Data:       {testing_rmse:.2f}")
print(f"SMAPE of Training Data:     {training_smape * 100:.2f}%")
print(f"SMAPE of Testing Data:      {testing_smape * 100:.2f}%")