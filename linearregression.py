X = df[['Temperature']]

y = df['Revenue']

## 4. Import function

from sklearn.linear_model import LinearRegression

## 5. Activate function

lr = LinearRegression()

## 6 Train Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state = 252)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

## 6. Fit model

lr.fit(X_train,y_train)

## 7. Predict

y_pred = lr.predict(X_test)

## 8. Parameters

lr.coef_ # Slope

lr.intercept_ # Intercept

# Icecream Revenue (y) = 44.2 + 21.4*Temperature(X)

## 9. Model Accuracy

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mean_absolute_error(y_test, y_pred)

mean_squared_error(y_test, y_pred)

r2_score(y_test,y_pred)

## Do future prediction

X_tomorrow = [[30]]

y_tomorrow = lr.predict(X_tomorrow)

y_tomorrow

## Visualize our model

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter(X,y)
ax.plot(X,lr.predict(X), color='red')
