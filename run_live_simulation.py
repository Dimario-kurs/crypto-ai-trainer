# === run_live_simulation.py ===
import ccxt
import torch
import torch.nn as nn
import numpy as np
import joblib
import time

# === 1. Модель ===
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

# === 2. Загрузка модели и scaler ===
scaler = joblib.load("scaler.pkl")

dummy_input_size = 23  # количество фичей (24 - time - y)
model = Net(input_size=dummy_input_size)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# === 3. Подключение к бирже Binance ===
exchange = ccxt.binance()
symbol = "BTC/USDT"
timeframe = "15m"

# === 4. Симуляция торговли ===
balance = 1000.0   # стартовый баланс USDT
trade_size = 10.0  # сумма на сделку
position = None    # текущая позиция (None, "long", "short")
entry_price = 0.0

print("🚀 Запуск симуляции...")

for i in range(96):  # примерно 1 сутки по 15м свечам
    # Загружаем свежие 100 свечей
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=100)
    last_candle = ohlcv[-1]  # [timestamp, open, high, low, close, volume]

    # Делаем фичи (пока только цена и объем)
    features = np.array([
        last_candle[1],  # open
        last_candle[2],  # high
        last_candle[3],  # low
        last_candle[4],  # close
        last_candle[5]   # volume
    ], dtype=np.float32).reshape(1, -1)

    # Нормализуем
    features = scaler.transform(features)

    # Прогноз
    x_tensor = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        pred = model(x_tensor)
        action = pred.argmax(1).item()  # 0=SELL, 1=HOLD, 2=BUY (с учетом сдвига)

    price = last_candle[4]

    # === Логика сделок ===
    if action == 2 and position is None:  # BUY
        position = "long"
        entry_price = price
        balance -= trade_size
        print(f"[{i}] Покупка по {price:.2f}, баланс={balance:.2f}")
    elif action == 0 and position == "long":  # SELL
        profit = trade_size * (price / entry_price)
        balance += profit
        print(f"[{i}] Продажа по {price:.2f}, прибыль={profit-trade_size:.2f}, баланс={balance:.2f}")
        position = None
    else:
        print(f"[{i}] Держим позицию ({'нет' if position is None else position}), цена={price:.2f}")

    time.sleep(1)  # имитация ожидания новой свечи (в реале можно оставить 60*15)

print(f"🏁 Симуляция завершена. Итоговый баланс={balance:.2f} USDT")



