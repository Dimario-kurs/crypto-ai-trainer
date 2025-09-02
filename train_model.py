import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler
import json
import datetime

# === 1. Загружаем данные ===
df = pd.read_csv("BTC_ETH_15m_features.csv")

# Отделяем признаки и целевую переменную
X = df.drop(columns=["time", "y"]).values.astype(np.float32)
y = df["y"].values.astype(np.int64)

# Масштабируем признаки
scaler = StandardScaler()
X = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl")

# === 2. Train/Test split ===
dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

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
losses, accs_train, accs_test = [], [], []
best_acc = 0
patience, patience_counter = 3, 0

log_rows = []

for epoch in range(30):  # максимум 30 эпох
    model.train()
    total_loss, correct = 0, 0
    for xb, yb in train_loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (pred.argmax(1) == yb).sum().item()
    acc_train = correct / len(train_ds)

    # Оценка на тесте
    model.eval()
    correct_test = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model(xb)
            correct_test += (pred.argmax(1) == yb).sum().item()
    acc_test = correct_test / len(test_ds)

    # Лог
    print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, Train Acc={acc_train:.2f}, Test Acc={acc_test:.2f}")
    losses.append(total_loss)
    accs_train.append(acc_train)
    accs_test.append(acc_test)
    log_rows.append([epoch+1, total_loss, acc_train, acc_test])

    # Early stopping
    if acc_test > best_acc:
        best_acc = acc_test
        patience_counter = 0
        torch.save(model.state_dict(), "model_best.pth")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("⏹ Early stopping triggered")
            break

# === 5. Визуализация ===
plt.plot(losses, label="Loss")
plt.plot(accs_train, label="Train Acc")
plt.plot(accs_test, label="Test Acc")
plt.legend()
plt.savefig("training_curve.png")
print("✅ Training finished, curve saved to training_curve.png")

# === 6. Сохранение модели ===
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
torch.save(model.state_dict(), f"model_{timestamp}.pth")
torch.save(model, f"model_full_{timestamp}.pth")
print(f"✅ Model saved to model_{timestamp}.pth and model_full_{timestamp}.pth")

# === 7. Лог в CSV ===
log_df = pd.DataFrame(log_rows, columns=["epoch", "loss", "train_acc", "test_acc"])
log_df.to_csv("training_log.csv", index=False)
print("✅ Training log saved to training_log.csv")

# === 8. Сохраняем конфиг ===
config = {
    "input_size": X.shape[1],
    "hidden": 64,
    "num_classes": 3,
    "batch_size": 64,
    "epochs": epoch+1,
    "best_test_acc": best_acc
}
with open("config.json", "w") as f:
    json.dump(config, f, indent=4)
print("✅ Config saved to config.json")

