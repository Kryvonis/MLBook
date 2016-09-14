import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('../datasets/wine/winequality-red.csv', sep=';')
print(df.describe())

plt.scatter(df['alcohol'], df['quality'])
plt.xlabel('Alcohol')
plt.ylabel('Quality')
plt.title('Alchol against quality')
plt.show()
