from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('../datasets/wine/winequality-red.csv', sep=';')
X = df[list(df.columns)[:-1]]
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X,y)

regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_prediction = regressor.predict(X_test)
print('R-quared : {}'.format(regressor.score(X_test,y_test)))