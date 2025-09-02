import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib

# === 1. Загружаем данные ===
df = pd.read_csv("BTC_ETH_15m_features.csv")

# Убираем колонку time
X = df.drop(columns=["time", "y"])

# Заполняем пропуски нулями
X = X.fillna(0)

# Нормализация признаков
scaler = StandardScaler()
X = scaler.fit_transform(X).astype(np.float32)

# Сохраняем нормализатор, чтобы использовать на новых данных
joblib.dump(scaler, "scaler.pkl")

# Метки: переводим {-1,0,1} → {0,1,2}
y = df["y"].replace({-1: 0, 0: 1, 1: 2}).astype(np.int64).values

# === 2. Dataset / DataLoader ===
dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# === 3. Модель ===
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

model = Net(X.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# === 4. Тренировка ===
losses, accs = [], []
for epoch in range(10):  # 10 эпох для примера
    total_loss, correct = 0, 0
    for xb, yb in loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (pred.argmax(1) == yb).sum().item()
    acc = correct / len(dataset)
    losses.append(total_loss)
    accs.append(acc)
    print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, Accuracy={acc:.2f}")

# === 5. Визуализация ===
plt.plot(losses, label="Loss")
plt.plot(accs, label="Accuracy")
plt.legend()
plt.savefig("training_curve.png")
print("✅ Training finished, curve saved to training_curve.png")
