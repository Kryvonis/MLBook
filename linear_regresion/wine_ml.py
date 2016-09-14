import numpy as np
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston

data = load_boston()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)



X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train = X_scaler.fit_transform(X_train)
y_train = y_scaler.fit_transform(y_train)

X_test = X_scaler.transform(X_test)
y_test = y_scaler.fit_transform(y_test)

# X_train = X_train.reshape(1,-1)
# X_test = X_test.reshape(1,-1)
# y_train = y_train.reshape(1,-1)
# y_test = y_test.reshape(1,-1)

regressor = SGDRegressor(loss='squared_loss')
scores = cross_val_score(regressor, X_train, y_train, cv=5)
print('Cross validation r-squared scores {}'.format(scores))
print('average cross validation r-squared scores {}'.format(np.mean(scores)))
regressor.fit_transform(X_train,y_train)
print('Test set r-squared score {}'.format(regressor.score(X_test,y_test)))