import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Data is taken from: https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification
# Some example codes that helped me to create this model: https://www.kaggle.com/code/dansbecker/your-first-machine-learning-model/tutorial

train_data = pd.read_csv('train.csv')
print("\nThe train data looks like this:\n")
print(train_data, "\n")

# Dropping any unnecessary values
train_data = train_data.dropna(axis=0)

# y variable is the price range for our model function
y = train_data.price_range

# Features of the phone to be considered in our model prediction
phone_features = ['battery_power', 'clock_speed', 'n_cores', 'ram', 'three_g'] # edit: if we change to talk_time instead of three_g the accuracy decrease

# New data set for our phones with the specified phone features
X = train_data[phone_features]

print("\nThe train data looks like this after considering 5 features:\n")
print(X, "\n")

# Defining the model and specifying a number for random_state to ensure same results each run
model = DecisionTreeRegressor(random_state=1)

# Fitting the model
model.fit(X, y)

# Getting the targeted data to find out if our model can predict the phone's price
target_data = pd.read_csv('target.csv')
print("\nThe target data looks like this:\n")
print(target_data, "\n")

# Dropping any unnecessary values
target_data = target_data.dropna(axis=0)

# Features of the phone to be considered in our model prediction
phone_features2 = ['battery_power', 'clock_speed', 'n_cores', 'ram', 'three_g']

# New data set for our target phones with the specified phone features
target = target_data[phone_features2]

print("Making predictions for the following phones: \n")
print(target, '\n')
# Price range: 0 - low cost, 1 - medium cost, 2 - high cost, 3 - very high cost
print("The real prices of the target phones are: ")
for price in target_data.price_range:
    print(price, end='. ')
print("\nThe predictions of our model are: ")
model_prediction = model.predict(target)
print(model_prediction, '\n')

total = 0
accurate = 0
for i, price in enumerate(target_data.price_range):
    if int(price) == int(model_prediction[i]):
        accurate += 1
    total += 1

print("Our model's accuracy with the trained dataset: ")
print(str(round(float((accurate/total)*100), 2))+"%\n")
