import ccxt
import torch
import torch.nn as nn
import pandas as pd
import json
import time
from collections import deque

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
input_size = X.shape[1]  # количество признаков

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

# История сделок
trades = []

# === 6. Основной цикл симуляции ===
while True:
    # Загружаем последние свечи
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=1)
    last = ohlcv[-1]
    _, open_, high, low, close, volume = last

    # Формируем фичи (очень упрощённо: только цена закрытия)
    features = torch.tensor([[close] * input_size], dtype=torch.float32)

    # Получаем прогноз
    with torch.no_grad():
        pred = model(features).argmax(1).item()

    # Торговая логика
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

    time.sleep(10)  # ждём 10 секунд перед новой свечой (для реального — ставим 60 * 15)


