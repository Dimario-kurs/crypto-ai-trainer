import ccxt
import torch
import torch.nn as nn
import pandas as pd
import json
import time
import csv
import os

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

# === 3. Определяем input_size из датасета ===
df = pd.read_csv("BTC_ETH_15m_features.csv")
X = df.drop(columns=["time", "y"]).values.astype("float32")
input_size = X.shape[1]

print(f"[INFO] Автоматически определён input_size = {input_size}")

# === 4. Загружаем модель ===
model = Net(input_size)
state_dict = torch.load("model.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# === 5. Настройки торговли ===
exchange = getattr(ccxt, config["exchange"])()
symbol = config["symbol"]
timeframe = config["timeframe"]
balance = config["initial_balance"]
trade_size = config["trade_size"]

# === 6. Инициализация ===
open_position = None   # (buy_price) если сделка открыта
log_file = "trades_log.csv"

# создаём лог, если его нет
if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "action", "price", "profit", "balance"])

# === 7. Основной цикл симуляции ===
while True:
    try:
        # загружаем последнюю свечу
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=1)
        last = ohlcv[-1]
        ts, open_, high, low, close, volume = last

        # (заглушка) формируем фичи
        features = torch.tensor([[close] * input_size], dtype=torch.float32)

        # прогноз
        with torch.no_grad():
            pred = model(features).argmax(1).item()

        action = "HOLD"
        profit = 0.0

        # BUY
        if pred == 1 and open_position is None and balance >= trade_size:
            balance -= trade_size
            open_position = close
            action = "BUY"

        # SELL
        elif pred == 2 and open_position is not None:
            buy_price = open_position
            profit = (close - buy_price) / buy_price * trade_size
            balance += trade_size + profit
            action = f"SELL (p={profit:.2f})"
            open_position = None

        # лог
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Close={close}, Action={action}, Balance={balance:.2f}")

        # записываем в CSV
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), action, close, profit, balance])

        time.sleep(10)  # для теста — каждые 10 сек, в реале ставим 60*15

    except Exception as e:
        print(f"[ERROR] {e}")
        time.sleep(30)
