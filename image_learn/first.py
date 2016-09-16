
from sklearn import datasets
digit = datasets.load_digits()
print('Digit - {}'.format(digit.target[0]))
print(digit.images[0])
print('Feature vector\n {} '.format(digit.images[0].reshape(-1,64)))