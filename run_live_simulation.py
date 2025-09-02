import ccxt
import torch
import torch.nn as nn
import joblib
import pandas as pd
import numpy as np


# === Архитектура сети (такая же, как при обучении) ===
class Net(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=3):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# === Загружаем модель и скейлер ===
scaler = joblib.load("scaler.pkl")

input_size = 4  # используем ["открыть", "высокий", "низкий", "закрывать"]
model = Net(input_size)  # создаем пустую модель
state_dict = torch.load("model.pth", map_location=torch.device("cpu"))
model.load_state_dict(state_dict)  # загружаем веса
model.eval()  # переводим в режим инференса


# === Подключение к Binance ===
exchange = ccxt.binance({
    "enableRateLimit": True,
    "options": {"defaultType": "spot"}
})

symbol = "BTC/USDT"
timeframe = "15m"
limit = 200  # количество последних свечей для симуляции


# === Функция получения свечей ===
def get_ohlcv():
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts", "открыть", "высокий", "низкий", "закрывать", "объем"])
    return df


# === Функция предсказания сигнала ===
def предсказать_сигнал(row):
    features = np.array([[row["открыть"], row["высокий"], row["низкий"], row["закрывать"]]])
    features_scaled = scaler.transform(features)
    tensor = torch.tensor(features_scaled, dtype=torch.float32)
    with torch.no_grad():
        output = model(tensor)
        signal = torch.argmax(output).item()
    # 0 = держать, 1 = long, 2 = short
    if signal == 1:
        return 1
    elif signal == 2:
        return -1
    else:
        return 0


# === Параметры симуляции ===
начальный_баланс = 1000.0
размер_торговли = 10.0
баланс = начальный_баланс
торги = []


# === Основной цикл симуляции ===
df = get_ohlcv()

for i in range(1, len(df)):
    row = df.iloc[i]
    сигнал = предсказать_сигнал(row)

    прибыль = 0
    if сигнал == 1:  # long
        прибыль = размер_торговли * ((df.iloc[i]["закрывать"] - df.iloc[i]["открыть"]) / df.iloc[i]["открыть"])
    elif сигнал == -1:  # short
        прибыль = размер_торговли * ((df.iloc[i]["открыть"] - df.iloc[i]["закрывать"]) / df.iloc[i]["открыть"])

    баланс += прибыль

    торги.append({
        "время": df.iloc[i]["ts"],
        "открыть": df.iloc[i]["открыть"],
        "закрывать": df.iloc[i]["закрывать"],
        "сигнал": сигнал,
        "прибыль": прибыль,
        "баланс": баланс
    })


# === Итоговые результаты ===
print(f"Начальный баланс: {начальный_баланс} USDT")
print(f"Финальный баланс: {баланс:.2f} USDT")
print(f"Сделок: {len(торги)}")

print("\nПоследние 5 сделок:")
for t in торги[-5:]:
    print(t)


# === Сохраняем историю ===
results_df = pd.DataFrame(торги)
results_df.to_csv("simulation_results.csv", index=False)
print("✅ История сохранена в simulation_results.csv")

