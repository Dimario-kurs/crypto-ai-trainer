# === run_live_simulation.py ===
import ccxt
import torch
import torch.nn as nn
import pandas as pd
import json
import time
import joblib
from collections import deque

# === 1. Загружаем config и scaler ===
with open("config.json", "r") as f:
    config = json.load(f)

scaler = joblib.load("scaler.pkl")

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

# === 3. Загружаем модель ===
input_size = config["input_size"]
model = Net(input_size, hidden=config["hidden"], num_classes=config["num_classes"])
state_dict = torch.load("model.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# === 4. Настройки торговли ===
exchange = getattr(ccxt, config.get("exchange", "binance"))()
symbol = config.get("symbol", "BTC/USDT")
timeframe = config.get("timeframe", "15m")
balance = config.get("initial_balance", 1000)
trade_size = config.get("trade_size", 10)

# История сделок
trades = []

print(f"[INFO] Запуск симуляции: {symbol}, {timeframe}, стартовый баланс = {balance} USDT")

# === 5. Основной цикл симуляции ===
while True:
    try:
        # Загружаем последнюю свечу
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=1)
        last = ohlcv[-1]
        _, open_, high, low, close, volume = last

        # === Формируем фичи ===
        # В live режиме у нас нет всех индикаторов, поэтому используем close и дублируем до input_size
        raw_features = [[close] * input_size]

        # Применяем scaler, чтобы признаки совпадали с обучением
        features = scaler.transform(raw_features)
        features = torch.tensor(features, dtype=torch.float32)

        # === Прогноз модели ===
        with torch.no_grad():
            pred = model(features).argmax(1).item()

        # === Торговая логика ===
        action = "HOLD"
        if pred == 1 and balance >= trade_size:  # BUY сигнал
            balance -= trade_size
            trades.append(("BUY", close))
            action = "BUY"
        elif pred == 2 and trades:  # SELL сигнал
            buy_price = trades[-1][1]
            profit = (close - buy_price) / buy_price * trade_size
            balance += trade_size + profit
            trades.append(("SELL", close, profit))
            action = f"SELL (profit={profit:.2f})"

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Close={close}, Action={action}, Balance={balance:.2f}")

        # Сохраняем сделки в CSV
        df_trades = pd.DataFrame(trades, columns=["action", "price", "profit"])
        df_trades.to_csv("trades_log.csv", index=False)

    except Exception as e:
        print(f"[ERROR] {e}")

    # Для теста ставим паузу 10 сек, в реальном режиме лучше 60*15 = 900 сек
    time.sleep(10)
