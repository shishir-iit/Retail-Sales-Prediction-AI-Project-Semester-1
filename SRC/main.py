import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv('Online_Retail.csv')
print('Shape:', df.shape)
print(df.isnull().sum())
print(df.head())

df = df.drop_duplicates()
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
df = df[df['StockCode'].astype(str).str.match(r'^\d')]     
df['Sales'] = df['Quantity'] * df['UnitPrice']
df = df[df['Sales'] < df['Sales'].quantile(0.9999)]          
print('Rows after cleaning:', len(df))

daily = (df.groupby(df['InvoiceDate'].dt.normalize())['Sales'].sum()
           .asfreq('D', fill_value=0))
daily = daily.iloc[:-1]            # last day is incomplete
print(daily.describe())

plt.figure(figsize=(12,4))
plt.plot(daily.index, daily.values)
plt.title('Daily Sales'); plt.xlabel('Date'); plt.ylabel('Sales (GBP)')
plt.grid(alpha=.25); plt.show()
