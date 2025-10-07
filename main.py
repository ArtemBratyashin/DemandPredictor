import pandas as pd
import numpy as np
import xgboost as xgb

df = pd.read_excel('Data.xlsx', sheet_name='Данные')
df['Время'] = pd.to_datetime(df['Время'])

period = 12
df = df.reset_index(drop=True)
df['month_index'] = np.arange(len(df))
df['sin_season'] = np.sin(2 * np.pi * df['month_index'] / period)
df['cos_season'] = np.cos(2 * np.pi * df['month_index'] / period)

for col in df.columns[1:]:
    df[f'{col}_lag1'] = df[col].shift(1)

df = df.dropna().reset_index(drop=True)

feature_cols = [col for col in df.columns if '_lag1' in col]

X = df[feature_cols]
Y = df['Целевые сделки']

X_train = X.iloc[:-1]
Y_train = Y.iloc[:-1]
X_pred = X.iloc[[-1]]

model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
model.fit(X_train, Y_train)

forecast = model.predict(X_pred)[0]
actual = Y.iloc[-1]
mape = abs(forecast - actual) / actual * 100

print(f'Прогноз на {df["Время"].iloc[-1].strftime("%Y-%m")}: {forecast:.0f}')
print(f'Фактическое значение: {actual}')
print(f'MAPE: {mape:.2f}%')