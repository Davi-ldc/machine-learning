import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('dataDL\BTC-USD.csv')
plt.plot(data['Open'])
plt.show()