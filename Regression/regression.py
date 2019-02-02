import pandas as pd
import quandl
import math,datetime
import numpy as np
from sklearn import preprocessing,svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
df=quandl.get('WIKI/GOOGL')
df1=df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df1['HL_PCT']=(df1['Adj. High']-df1['Adj. Close']) / df1['Adj. Close'] * 100.0
df1['HL_change']=(df1['Adj. Close'] - df1['Adj. Open']) / df1['Adj. Open'] * 100.0
df2=df1[['Adj. Close','HL_PCT','HL_change','Adj. Volume']]
forecast_col = 'Adj. Close'
df2.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01 * len(df2)))
# print(forecast_out)
df2['label'] = df2[forecast_col].shift(-forecast_out)

X = np.array(df2.drop(['label'], 1))
X = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately = X[-forecast_out:]

df2.dropna(inplace=True)
y = np.array(df2['label'])
y = np.array(df2['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)

df2['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df2.loc[next_date] = [np.nan for _ in range(len(df2.columns) - 1)] + [i]

df2['Adj. Close'].plot()
df2['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()