import torch
import os
import torch.nn as nn
import torch.optim as optim
import time

# ================== НАСТРОЙКИ ==================
MODEL_PATH = "solar_shield.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.005

# ================== МОДЕЛЬ ==================
class SolarShieldAI(nn.Module):
    def __init__(self, input_size=2, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

# ================== ИНИЦИАЛИЗАЦИЯ ==================
model = SolarShieldAI().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print("Модель загружена")

# ================== ОБУЧЕНИЕ (ОДИН ШАГ) ==================
def train_step(x, y):
    model.train()
    optimizer.zero_grad()

    prediction = model(x)
    loss = criterion(prediction, y)

    loss.backward()
    optimizer.step()

    return loss.item(), prediction.item()

# ================== ЖИВОЙ ЦИКЛ ==================
print("\n--- ЗАПУСК САМООБУЧАЮЩЕЙСЯ СИСТЕМЫ ---")
print("\n--- ВВЕДИТЕ \"Y\" ЕСЛИ ХОТИТЕ ОБУЧИТЬ ИЛИ \"N\" ЕСЛИ ИСПОЛЬЗОВАТЬ ---")
choice = input()

if choice == "Y":
    try:
        with open("data.learn", "r") as data:
            for line in data:
                if not line.strip():
                    continue

                try:
                    v1, v2, actual = map(float, line.split())
                except ValueError:
                    print("Пропуск битой строки:", line.strip())
                    continue

                # Нормализация (пример)
                v1 /= 100.0
                v2 /= 100.0

                state = torch.tensor(
                    [[[v1, v2]]],
                    dtype=torch.float32,
                    device=DEVICE
                )
                target = torch.tensor([[actual]], dtype=torch.float32, device=DEVICE)

                # Прогноз
                model.eval()
                with torch.no_grad():
                    predicted = model(state).item()

                # Обучение
                loss, trained_pred = train_step(state, target)

                print(f"""
    [ПРОГНОЗ ИИ]    : {predicted:.4f}
    [РЕАЛЬНОСТЬ]   : {actual:.4f}
    [ПОСЛЕ ОБУЧ.]  : {trained_pred:.4f}
    [ОШИБКА]       : {loss:.6f}
    --------------------------------
    """)

    except KeyboardInterrupt:
        print("\nПрерывание пользователем")

    finally:
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }, MODEL_PATH)

        print("Состояние ИИ сохранено")
else:
    print("\n--- РЕЖИМ ИСПОЛЬЗОВАНИЯ ИИ ---")
    print("Введите два параметра (например: v1 v2)")
    print("Для выхода введите Ctrl+C\n")

    try:
        while True:
            user_input = input("Ввод: ").strip()
            if not user_input:
                continue

            try:
                v1, v2 = map(float, user_input.split())
            except ValueError:
                print("Ошибка ввода. Используйте формат: v1 v2")
                continue

            # Нормализация
            v1 /= 100.0
            v2 /= 100.0

            state = torch.tensor(
                [[[v1, v2]]],
                dtype=torch.float32,
                device=DEVICE
            )

            # Прогноз без обучения
            model.eval()
            with torch.no_grad():
                prediction = model(state).item()

            print(f"""
    [ПРОГНОЗ ИИ] : {prediction:.4f}
    -------------------------------
    """)

    except KeyboardInterrupt:
        print("\nВыход из режима использования")
