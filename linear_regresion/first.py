# end in 35 p.

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

X = [[6, 2], [8, 1], [10, 0], [14, 2], [18, 0]]
y = [[7], [9], [13], [17.5], [18]]
X_test = [[8, 2], [9, 0], [11, 2], [16, 2], [12, 0]]
y_test = [[11], [8.5], [15], [18], [11]]

model = LinearRegression()
model.fit(X, y)
prediction = model.predict(X_test)

for i, prediction in enumerate(prediction):
    print('Predicted {}, Target {}'.format(prediction, y_test[i]))
print('R-squared: {:.2} '.format(model.score(X_test, y_test)))





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
