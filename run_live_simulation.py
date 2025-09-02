import ccxt
import torch
import torch.nn as nn
import pandas as pd
import joblib
import json
import time
import csv

# === 1. Загружаем конфиг ===
with open("config.json", "r") as f:
    config = json.load(f)

# === 2. Класс модели ===
class Net(nn.Module):
    def __init__(self, input_size, hidden=64, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# === 3. Определяем input_size из CSV ===
df = pd.read_csv("BTC_ETH_15m_features.csv")
X = df.drop(columns=["time", "y"]).values.astype("float32")
input_size = X.shape[1]
print(f"[INFO] Автоматически определён input_size = {input_size}")

# === 4. Подгружаем scaler ===
scaler = joblib.load("scaler.pkl")

# === 5. Загружаем модель ===
model = Net(input_size)
state_dict = torch.load("model.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# === 6. Настройки торговли ===
exchange = getattr(ccxt, config["exchange"])()
symbol = config["symbol"]
timeframe = config["timeframe"]
balance = config["initial_balance"]
trade_size = config["trade_size"]

trades = []

# === 7. Лог сделок в CSV ===
with open("trades_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time", "action", "price", "balance", "profit"])

# === 8. Основной цикл симуляции ===
while True:
    try:
        # Загружаем последнюю свечу
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=1)
        _, open_, high, low, close, volume = ohlcv[-1]

        # Формируем фичи
        features = [[close] * input_size]
        features = scaler.transform(features)
        features = torch.tensor(features, dtype=torch.float32)

        # Прогноз модели
        with torch.no_grad():
            pred = model(features).argmax(1).item()

        action = "HOLD"
        profit = 0.0

        if pred == 1 and balance >= trade_size:  # BUY
            balance -= trade_size
            trades.append(("BUY", close))
            action = "BUY"

        elif pred == 2 and trades and trades[-1][0] == "BUY":  # SELL
            buy_price = trades[-1][1]
            profit = (close - buy_price) / buy_price * trade_size
            balance += trade_size + profit
            trades.append(("SELL", close, profit))
            action = f"SELL (profit={profit:.2f})"

        now = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{now}] Close={close}, Action={action}, Balance={balance:.2f}")

        # Лог в CSV
        with open("trades_log.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([now, action, close, balance, profit])

    except Exception as e:
        print("❌ Ошибка:", e)

    time.sleep(10)  # проверка каждые 10 секунд
£

