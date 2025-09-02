import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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

# Сохраняем нормализатор
joblib.dump(scaler, "scaler.pkl")

# Метки: переводим {-1,0,1} → {0,1,2}
y = df["y"].replace({-1: 0, 0: 1, 1: 2}).astype(np.int64).values

# === 2. Делим на train/test ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_ds  = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=64)

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
train_losses, test_accs, train_accs = [], [], []

for epoch in range(10):  # можно увеличить до 50+
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
    correct_test = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model(xb)
            correct_test += (pred.argmax(1) == yb).sum().item()
    test_acc = correct_test / len(test_ds)

    # Логируем
    train_losses.append(total_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    print(f"Epoch {epoch+1}: Train Loss={total_loss:.4f}, Train Acc={train_acc:.2f}, Test Acc={test_acc:.2f}")

# === 5. Визуализация ===
plt.figure(figsize=(10,5))
plt.plot(train_losses, label="Train Loss")
plt.plot(train_accs, label="Train Accuracy")
plt.plot(test_accs, label="Test Accuracy")
plt.legend()
plt.savefig("training_curve.png")
print("✅ Training finished, curve saved to training_curve.png")

# Сохраняем state_dict (рекомендуемый вариант)
torch.save(model.state_dict(), "model.pth")

# Дополнительно сохраняем всю модель (если захочешь загружать напрямую)
torch.save(model, "model_full.pth")

print("✅ Model saved to model.pth (state_dict) and model_full.pth (full model)")

