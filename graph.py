import numpy as np
import matplotlib.pyplot as plt

# ====== Ввод данных ======
v_ship = float(input("Скорость корабля (км/с): "))
distance = float(input("Расстояние до опасной зоны (км): "))

print("Тип угрозы:")
print("1 — Солнечная вспышка")
print("2 — Поток заряженных частиц")
print("3 — Микрометеориты")
threat_type = int(input("Введите номер угрозы: "))

intensity = float(input("Интенсивность угрозы (0–1000): "))

# ====== Параметры угроз ======
if threat_type == 1:
    threat_name = "Солнечная вспышка"
    base_plasma = 75
    base_field = 6
elif threat_type == 2:
    threat_name = "Заряженные частицы"
    base_plasma = 55
    base_field = 4.5
else:
    threat_name = "Микрометеориты"
    base_plasma = 35
    base_field = 3

# ====== Время до столкновения ======
collision_time = distance / v_ship
reaction_time = max(0.8, 18 / v_ship)

# ====== ИИ выбирает параметры плазмы и поля ======
plasma_density = min(100, base_plasma + intensity / 15)
B_field = min(8, base_field + intensity / 250)

bubble_radius = 1 + plasma_density / 80
power = 0.35 * (B_field ** 2) * bubble_radius

# ====== ЭФФЕКТИВНОСТЬ ЗАЩИТЫ ======
protection_factor = min(1, power / (intensity + 1))
time_factor = min(1, reaction_time / collision_time)


# ====== КОРРЕКЦИЯ ТРАЕКТОРИИ ======
angle_change = min(45, intensity / 22)  # угол отклонения в градусахx

time = np.linspace(0, reaction_time, 150)
x_before = time * v_ship
y_before = np.zeros_like(time)

x_after = time * v_ship
y_after = np.tan(np.radians(angle_change)) * x_after

# ====== Плазменный пузырь (асимметричный) ======
theta = np.linspace(0, 2*np.pi, 400)
asymmetry = min(intensity / 1200, 0.9)
speed_effect = min(v_ship / 40, 0.6)

radius = bubble_radius + asymmetry * np.cos(theta) - speed_effect * np.sin(theta)
bubble_x = radius * np.cos(theta)
bubble_y = radius * np.sin(theta)

# ====== График ======
plt.figure(figsize=(8,8))
plt.plot(x_before, y_before, '--', label="Без защиты")
plt.plot(x_after, y_after, label=f"С ИИ-защитой, угол {angle_change:.1f}°")
plt.plot(bubble_x, bubble_y, color='purple', linewidth=2, label="Плазменно-магнитный пузырь")
plt.scatter(0, 0, color='black', s=40, label="Корабль")

plt.title("ИИ-управляемая плазменно-магнитная защита")
plt.xlabel("Пространство (км)")
plt.ylabel("Отклонение (км)")
plt.legend()
plt.axis("equal")
plt.grid(True)
plt.show()

# ====== Вывод результатов ======
print("\n--- РЕШЕНИЕ ИИ ---")
print(f"Тип угрозы: {threat_name}")
print(f"Время до опасной зоны: {collision_time:.1f} сек")
print(f"Время реакции ИИ: {reaction_time:.2f} сек")
print(f"Выбранная плотность плазмы: {plasma_density:.1f}%")
print(f"Магнитное поле: {B_field:.2f} Тл")
print(f"Мощность катушек: {power:.2f} МВт")
print(f"Угол отклонения: {angle_change:.1f}°")
