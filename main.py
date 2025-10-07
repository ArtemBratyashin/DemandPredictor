import pandas as pd
import numpy as np
import xgboost as xgb

# Загружаем лист данных (первая колонка — дата, остальные — признаки, последний — целевая переменная)
df = pd.read_excel('Data.xlsx', sheet_name='Данные')
df['Время'] = pd.to_datetime(df['Время'])  # первая колонка — дата

period = 12
df = df.reset_index(drop=True)
df['month_index'] = np.arange(len(df))
df['sin_season'] = np.sin(2 * np.pi * df['month_index'] / period)
df['cos_season'] = np.cos(2 * np.pi * df['month_index'] / period)

# Добавляем лаговые признаки для всех признаков кроме даты и целевой переменной
for col in df.columns[1:-1]:  # пропускаем дату и целевой столбец
    df[f'{col}_lag1'] = df[col].shift(1)

df = df.dropna().reset_index(drop=True)

# Формируем список признаков:
feature_cols = [col for col in df.columns if '_lag1' in col or col in ['sin_season', 'cos_season']]
target_col = df.columns[-1]  # последний столбец — целевой

X = df[feature_cols]
y = df[target_col]

X_train = X[:-1]
y_train = y[:-1]
X_pred = X[-1:]

model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

forecast = model.predict(X_pred)[0]
actual = y.iloc[-1]
mape = abs(forecast - actual) / actual * 100

print(f'Прогноз на {df["col_0"].iloc[-1].strftime("%Y-%m")}: {forecast:.0f}')
print(f'Фактическое значение: {actual}')
print(f'MAPE: {mape:.2f}%')