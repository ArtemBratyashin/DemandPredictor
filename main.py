import xgboost as xgb
from src.csv_file import csv_file

df = (
    csv_file("data/prepared_data.csv")
    .load_df()
)

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