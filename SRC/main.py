import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

os.makedirs("../outputs", exist_ok=True)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "Data", "Online_Retail.csv")

df = pd.read_csv(DATA_PATH)
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
plt.grid(alpha=.25)
plt.savefig("../outputs/daily_sales.png", dpi=300, bbox_inches="tight")
print("Saved outputs/daily_sales.png")
plt.show()

f = pd.DataFrame({'Sales': daily})
f['target'] = f['Sales'].shift(-1)                 # tomorrow's sales
f['roll3']  = f['Sales'].rolling(3).mean()
f['roll7']  = f['Sales'].rolling(7).mean()
f['lag7']   = f['Sales'].shift(6)
next_dow    = (f.index.dayofweek + 1) % 7
f['next_sat'] = (next_dow == 5).astype(int)
f['next_sun'] = (next_dow == 6).astype(int)
f['q4']       = f.index.month.isin([10, 11, 12]).astype(int)
f = f.dropna()

feature_columns = ['Sales','roll3','roll7','lag7','next_sat','next_sun','q4']
X = f[feature_columns]
y = f['target']
print(X.shape)
print(f.head())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print('Train:', len(X_train), ' Test:', len(X_test))

model = LinearRegression()
model.fit(X_train, y_train)

coef = pd.DataFrame({'Feature': feature_columns, 'Coefficient': model.coef_})
print('Intercept:', round(model.intercept_, 2))
print(coef)

y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print("\n" + "="*45)
print("       ONLINE RETAIL SALES PREDICTION")
print("="*45)

print("Model: Linear Regression")
print(f"MAE  : {mae:,.2f}")
print(f"RMSE : {rmse:,.2f}")
print(f"R2   : {r2:.3f}")
plt.figure(figsize=(7,6))
plt.scatter(y_test, y_pred, alpha=.7, edgecolor='black')
lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
plt.plot([lo, hi], [lo, hi], '--', label='Perfect prediction')
plt.xlabel('Actual next-day sales'); plt.ylabel('Predicted next-day sales')
plt.title('Actual vs Predicted')
plt.legend()
plt.grid(alpha=.25)

plt.savefig("../outputs/actual_vs_predicted.png", dpi=300, bbox_inches="tight")
print("Saved outputs/actual_vs_predicted.png")
plt.show()

latest = X.iloc[[-1]]
print("\nFeatures used (last day in data):")
print(latest)

prediction = model.predict(latest)[0]

print(f"\nPredicted next-day sales: GBP {prediction:,.2f}")
print("\nAll outputs saved successfully!")
print("="*45)
