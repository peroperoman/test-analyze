import datetime
import IPython.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas_datareader
import sklearn
import sklearn.linear_model
import sklearn.model_selection

# yahooのapiで株価（アップル・フェイスブック・金）を取得
df_aapl = pandas_datareader.data.DataReader('AAPL', 'yahoo', '2014-11-01')
df_fb = pandas_datareader.data.DataReader('FB', 'yahoo', '2014-11-01')
df_gold = pandas_datareader.data.DataReader('GLD', 'yahoo', '2014-11-01')

# 統計
df_aapl[ 'SMA' ] = df_aapl[ 'Close' ].rolling(window=14).mean()
df_aapl[ 'Close' ].plot(figsize=(15,6), color='red')
df_aapl[ 'SMA' ].plot(figsize=(15,6), color='green')
plt.show()

# 値上がり率・値下がり率
df_aapl[ 'change' ] = (((df_aapl[ 'Close' ] - df_aapl[ 'Open' ])) / (df_aapl[ 'Open' ]) * 100)
df_fb[ 'change' ] = (((df_fb[ 'Close' ] - df_fb[ 'Open' ])) / (df_fb[ 'Open' ]) * 100)
df_gold[ 'change' ] = (((df_gold[ 'Close' ] - df_gold[ 'Open' ])) / (df_gold[ 'Open' ]) * 100)
df_aapl.tail(2).round(2)

# 傾向比較
df_aapl[ 'Close' ].plot(figsize=(15,6), color='red')
df_fb[ 'Close' ].plot(figsize=(15,6), color='blue')
plt.show()

df_aapl[ 'Close' ].plot(figsize=(15,6), color='red')
df_gold[ 'Close' ].plot(figsize=(15,6), color='orange')
plt.show()

df_aapl[ 'change' ].tail(100).plot(grid=True, figsize=(15,6), color='red')
df_fb[ 'change' ].tail(100).plot(grid=True, figsize=(15,6), color='blue')
df_gold[ 'change' ].tail(100).plot(grid=True, figsize=(15,6), color='orange')
plt.show()

# 機械学習
df_aapl[ 'label' ] = df_aapl[ 'Close' ].shift(-30)
df_aapl.tail(40)

x = np.array(df_aapl.drop([ 'label', 'SMA' ], axis=1))
x = sklearn.preprocessing.scale(x)
predict_data = x[-30:]
x = x[:-30]
y = np.array(df_aapl[ 'label' ])
y = y[:-30]
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    x, y, test_size = 0.2)

lr = sklearn.linear_model.LinearRegression()
lr.fit(x_train,y_train)

accuracy = lr.score(x_test, y_test)
#accuracy

predicted_data = lr.predict(predict_data)
#predicted_data

# 株価の予測値をplot
df_aapl[ 'Predict' ] = np.nan
last_data = df_aapl.iloc[-1].name
one_day = 86400
next_unix = last_data.timestamp() + one_day

for data in predicted_data:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df_aapl.loc[next_date] = np.append([np.nan]* (len(df_aapl.columns)-1), data)

df_aapl[ 'Close' ].plot(figsize=(15,6), color='green')
df_aapl[ 'Predict' ].plot(figsize=(15,6), color='orange')
plt.show()

