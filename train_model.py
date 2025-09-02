# === train_model.py ===
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
from sklearn.preprocessing import StandardScaler

# === 1. Загружаем данные ===
df = pd.read_csv("BTC_ETH_15m_features.csv")

# Убираем строки с NaN
df = df.dropna().reset_index(drop=True)
print(f"[INFO] После очистки осталось строк: {len(df)}")

# Отделяем признаки и целевую переменную
X = df.drop(columns=["time", "y"]).values.astype(np.float32)
y = df["y"].values.astype(np.int64)

# Сдвигаем метки {-1,0,1} → {0,1,2}
y = y + 1

# === 2. Нормализация признаков ===
scaler = StandardScaler()
X = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl")

# === 3. Dataset / DataLoader ===
dataset = TensorDataset(torch.tensor(X), torch.tensor(y))

# Разделим на train/test (80/20)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = random_split(dataset, [train_size, test_size],
                                 generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

# === 4. Модель ===
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

input_size = X.shape[1]
model = Net(input_size)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # сниженный lr
loss_fn = nn.CrossEntropyLoss()

# === 5. Обучение ===
epochs = 10
train_losses, test_accs = [], []
log_rows = []

for epoch in range(epochs):
    # --- Train ---
    model.train()
    total_loss, correct, count = 0, 0, 0
    for xb, yb in train_loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (pred.argmax(1) == yb).sum().item()
        count += len(yb)
    train_acc = correct / len(train_ds)
    avg_loss = total_loss / len(train_loader)

    # --- Test ---
    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model(xb)
            correct += (pred.argmax(1) == yb).sum().item()
    test_acc = correct / len(test_ds)

    train_losses.append(avg_loss)
    test_accs.append(test_acc)

    log_rows.append([epoch+1, avg_loss, train_acc, test_acc])
    print(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Train Acc={train_acc:.2f}, Test Acc={test_acc:.2f}")

# === 6. Сохранение модели и графика ===
torch.save(model.state_dict(), "model.pth")
plt.plot(train_losses, label="Train Loss")
plt.plot(test_accs, label="Test Accuracy")
plt.legend()
plt.savefig("training_curve.png")

# Сохраняем лог обучения
log_df = pd.DataFrame(log_rows, columns=["epoch", "train_loss", "train_acc", "test_acc"])
log_df.to_csv("training_log.csv", index=False)

# Сохраняем config.json (input_size пригодится в симуляции)
config = {
    "input_size": input_size,
    "hidden": 64,
    "num_classes": 3
}
with open("config.json", "w") as f:
    json.dump(config, f, indent=2)

print("✅ Training finished, curve saved to training_curve.png")
print("✅ Model saved to model.pth")
print("✅ Training log saved to training_log.csv")
print("✅ Config saved to config.json")
