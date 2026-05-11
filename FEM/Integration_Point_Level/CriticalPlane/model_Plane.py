import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # ДОБАВЛЕНО: импорт pandas для сохранения в Excel
import material as mt
import criterion as cr

# Настройки лучей
from tensor import StressTensor

NUM_RAYS = 120  # количество лучей на 360 градусов
MAX_DIST = 20.0  # максимальная дальность луча
STEP = 0.05  # шаг движения вдоль луча

# ДОБАВЛЕНО: Выносим угол в отдельную переменную, чтобы использовать ее и для расчетов, и для имени листа
CURRENT_ALPHA_DEG = 90

tensor_data = """
0.900000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
0.000000 0.399714 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
0.000000 0.000000 0.900000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
0.000000 0.000000 0.000000 0.399994 0.000000 0.000000 0.000000 0.000000 0.000000
0.000000 0.000000 0.000000 0.000000 0.396589 0.000000 0.000000 0.000000 0.000000
0.000000 0.000000 0.000000 0.000000 0.000000 0.396613 0.000000 0.000000 0.000000
0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.544270 0.000000 0.000000
0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.400018 0.000000
0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.900000
                """
A_matrix = mt.load_tensor_from_string(tensor_data)
material = mt.Material(
    mu=-0.7,
    A_tensor=A_matrix,
    Rpx=0.3, Rpy=0.3, Rpz=0.15,
    Rcx=10.0, Rcy=10.0, Rcz=8.5
)
my_material = material


def is_inside(x, y, alpha_deg=CURRENT_ALPHA_DEG):
    sigma_1 = x
    sigma_2 = y

    alpha = math.radians(alpha_deg)
    c = math.cos(alpha)
    s = math.sin(alpha)

    sigma_x = sigma_1 * c ** 2 + sigma_2 * s ** 2
    sigma_z = sigma_1 * s ** 2 + sigma_2 * c ** 2
    tau_xz = -(sigma_1 - sigma_2) * s * c

    stress_tensor = StressTensor(sigma_x, 0.0, sigma_z, 0.0, 0.0, tau_xz)
    f_shear, _, _ = cr.find_critical_plane_shear(stress_tensor, material, mode='2D_XZ')
    f_comp, _, _ = cr.find_critical_plane_compression(stress_tensor, material, mode='2D_XZ')
    f_ten, _, _ = cr.find_critical_plane_tensile(stress_tensor, material, mode='3D')
    overall_max_f = max(f_shear, f_comp, f_ten)

    return overall_max_f >= 0


# Сбор точек пересечения лучей с объектами
intersection_points = []

for i in range(NUM_RAYS):
    angle = 2 * math.pi * i / NUM_RAYS  # угол в радианах
    dx = math.cos(angle)
    dy = math.sin(angle)

    t = STEP
    found = False
    while t <= MAX_DIST:
        x = t * dx
        y = t * dy
        # ДОБАВЛЕНО: передаем текущий угол в функцию
        if is_inside(x, y, CURRENT_ALPHA_DEG):
            intersection_points.append((x, y))
            found = True
            break
        t += STEP

# --- ДОБАВЛЕННЫЙ БЛОК: СОХРАНЕНИЕ В EXCEL ---
if intersection_points:
    # 1. Создаем таблицу (DataFrame) из списка точек
    df = pd.DataFrame(intersection_points, columns=['sigma_1', 'sigma_2'])

    excel_filename = "Точки_разрушения.xlsx"
    sheet_name = str(CURRENT_ALPHA_DEG)  # Называем лист по номеру угла

    # 2. Логика сохранения.
    # Используем ExcelWriter в режиме 'a' (append), чтобы листы не перезаписывали
    # друг друга, если вы будете запускать скрипт для разных углов.
    import os

    if os.path.exists(excel_filename):
        with pd.ExcelWriter(excel_filename, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        df.to_excel(excel_filename, sheet_name=sheet_name, index=False)

    print(f"Точки успешно сохранены в файл '{excel_filename}' на лист '{sheet_name}'")
else:
    print("Точки пересечения не найдены, сохранять нечего.")
# ---------------------------------------------


# Визуализация
plt.figure(figsize=(8, 8))

# Рисуем сохранённые точки
if intersection_points:
    xs, ys = zip(*intersection_points)
    plt.scatter(xs, ys, c='red', s=30, label='Точки пересечения')

# Также можно нарисовать несколько лучей (каждый 4-й для наглядности)
for i in range(0, NUM_RAYS, max(1, NUM_RAYS // 20)):
    angle = 2 * math.pi * i / NUM_RAYS
    dx = math.cos(angle)
    dy = math.sin(angle)
    # Рисуем луч до максимальной дальности
    plt.plot([0, dx * MAX_DIST], [0, dy * MAX_DIST], color='gray', linewidth=0.5, alpha=0.5)

# Настройки графика
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True, linestyle='--', alpha=0.7)
plt.axis('equal')
plt.title(f'Лучи из начала координат ({NUM_RAYS} лучей), Угол = {CURRENT_ALPHA_DEG}°')
plt.legend()
plt.show()
