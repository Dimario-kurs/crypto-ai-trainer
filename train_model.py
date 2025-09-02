# === train_model.py ===
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler

# === 1. Загружаем данные ===
df = pd.read_csv("BTC_ETH_15m_features.csv")

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
train_ds, test_ds = random_split(dataset, [train_size, test_size])

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

model = Net(X.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# === 5. Обучение ===
epochs = 10
train_losses, test_accs = [], []

for epoch in range(epochs):
    # --- Train ---
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
    train_acc = correct / len(train_ds)

    # --- Test ---
    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model(xb)
            correct += (pred.argmax(1) == yb).sum().item()
    test_acc = correct / len(test_ds)

    train_losses.append(total_loss)
    test_accs.append(test_acc)

    print(f"Epoch {epoch+1}: Train Loss={total_loss:.4f}, Train Acc={train_acc:.2f}, Test Acc={test_acc:.2f}")

# === 6. Сохранение модели и графика ===
torch.save(model.state_dict(), "model.pth")
plt.plot(train_losses, label="Train Loss")
plt.plot(test_accs, label="Test Accuracy")
plt.legend()
plt.savefig("training_curve.png")

print("✅ Training finished, curve saved to training_curve.png")
print("✅ Model saved to model.pth")

