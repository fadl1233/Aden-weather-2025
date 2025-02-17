import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# تحميل البيانات بكفاءة
data = pd.read_excel(r"C:\Users\AL BASHA\PycharmProjects\w-a-2025\export.xlsx", usecols=['date', 'tavg'])

# معالجة البيانات
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)
data.dropna(inplace=True)

# تطبيع البيانات
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# إعداد البيانات للنموذج
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

seq_length = 30
X, y = create_sequences(data_scaled, seq_length)

# تقسيم البيانات
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# بناء النموذج بكفاءة
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(seq_length, X.shape[2])),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# تدريب النموذج
model.fit(X_train, y_train, epochs=15, batch_size=64, validation_data=(X_test, y_test))

# التنبؤ لعام 2025
forecast_days = 365
last_sequence = data_scaled[-seq_length:].copy()
forecast = []

for _ in range(forecast_days):
    pred = model.predict(last_sequence.reshape(1, seq_length, -1), verbose=0)
    forecast.append(pred[0, 0])
    last_sequence = np.vstack([last_sequence[1:], pred])

# عكس التطبيع
forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

# إنشاء جدول التنبؤ
forecast_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='D')
forecast_df = pd.DataFrame({'date': forecast_dates, 'forecast_tavg': forecast.flatten()})

# حفظ التنبؤ في ملف
forecast_df.to_excel(r"C:\Users\AL BASHA\PycharmProjects\w-a-2025\forecast_2025.xlsx", index=False)

# رسم التنبؤ
plt.figure(figsize=(14, 7))
plt.plot(forecast_df['date'], forecast_df['forecast_tavg'], label='Forecasted Temperature', color='red')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('2025 Temperature Forecast')
plt.legend()
plt.show()
