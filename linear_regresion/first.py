import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
X = [[6], [8], [10], [14], [18]]
y = [[7], [9], [13], [17.5], [18]]
X_test = [[8], [9], [11], [16], [12]]
y_test = [[11], [8.5], [15], [18], [11]]

model = LinearRegression()
model.fit(X,y)
X_predict = [12]
y_predict = model.predict(X_predict)[0]

print('A 12" pizza should cost: $%.2f' % model.predict([12])[0])

print('Residual sum of squares: %.2f' % np.mean((model.predict(X)- y) ** 2))

print('R-squared: %.4f' % model.score(X_test, y_test))

#################
#### Ploting ####
#################
#
# plt.figure()
# plt.title('Pizza price against diameter')
# plt.xlabel('Diameter in inches')
# plt.ylabel('Price in dollars')
# plt.plot(X, y, 'k.')
# plt.axis([0,20,0,20])
# plt.grid(True)
# plt.show()
