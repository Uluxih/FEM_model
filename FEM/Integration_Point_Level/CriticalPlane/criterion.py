import numpy as np
import pytest
from FEM.Integration_Point_Level.CriticalPlane.tensor import StressTensor
import FEM.Integration_Point_Level.CriticalPlane.material as mt


# ==========================================
# ДОБАВЛЕНО: Функция вычисления вектора нормали
# ==========================================
def get_normal_vector(params, mode='3D'):
    """
    Преобразует угловые параметры в единичный вектор нормали.
    В 3D: params = [theta, phi] (полярный и азимутальный углы)
    В 2D_XZ: params = theta (угол в плоскости XZ)
    """
    if mode == '3D':
        theta, phi = params[0], params[1]
        nx = np.sin(theta) * np.cos(phi)
        ny = np.sin(theta) * np.sin(phi)
        nz = np.cos(theta)
        return np.array([nx, ny, nz])
    elif mode == '2D_XZ':
        # Обработка случая, когда params передается как список [theta] или как число theta
        theta = params[0] if isinstance(params, (list, tuple, np.ndarray)) else params
        nx = np.sin(theta)
        ny = 0.0
        nz = np.cos(theta)
        return np.array([nx, ny, nz])
    else:
        raise ValueError(f"Неизвестный режим: {mode}")


# ==========================================
# НОВЫЙ БЛОК: ГЕНЕРАЦИЯ ДИСКРЕТНЫХ ПЛОЩАДОК (ОПТИМИЗАЦИЯ)
# ==========================================
def generate_discrete_params(num_planes, mode='3D'):
    """
    Генерирует равномерно распределенный набор параметров (углов) для поиска.
    Для 3D используется алгоритм сферы Фибоначчи для равномерного покрытия сферы.
    Для 2D_XZ - равномерное распределение по полуокружности.
    """
    params_list = []
    if mode == '3D':
        if num_planes == 1:
            return [[0.0, 0.0]]

        phi_golden = np.pi * (3.0 - np.sqrt(5.0))  # Золотой угол в радианах
        for i in range(num_planes):
            # y изменяется от 1 до -1
            y = 1.0 - (i / float(num_planes - 1)) * 2.0
            # Защита от ошибок округления для arccos
            y = max(-1.0, min(1.0, y))

            theta = np.arccos(y)
            phi = (i * phi_golden) % (2 * np.pi)
            params_list.append([theta, phi])

    elif mode == '2D_XZ':
        for i in range(num_planes):
            theta = i * np.pi / num_planes
            params_list.append([theta])
    else:
        raise ValueError(f"Неизвестный режим: {mode}")

    return params_list


# ==========================================
# БЛОК: ОТДЕЛЬНЫЕ ФУНКЦИИ ПРЕДЕЛОВ ПРОЧНОСТИ
# ==========================================
def get_tensile_limit(n, material: mt.Material):
    """Вычисляет предел прочности на растяжение на заданной площадке"""
    return material.Rpx * n[0] ** 2 + material.Rpy * n[1] ** 2 + material.Rpz * n[2] ** 2


def get_compression_limit(n, material: mt.Material):
    """Вычисляет предел прочности на сжатие на заданной площадке"""
    return material.Rcx * n[0] ** 2 + material.Rcy * n[1] ** 2 + material.Rcz * n[2] ** 2


def get_cohesion_limit(n, stress_tensor: StressTensor, material: mt.Material):
    """Вычисляет предельное сцепление (C) на площадке с учетом направления сдвига"""
    tau_n = stress_tensor.shear_stress_magnitude(n)

    # Направление сдвига (s_geom) – единичный вектор в плоскости площадки
    if tau_n > 1e-9:
        s_geom = stress_tensor.shear_stress_vector(n) / tau_n
    else:
        # Выбираем произвольное направление, ортогональное нормали
        if abs(n[2]) < 0.9:
            s_geom = np.cross(n, np.array([0, 0, 1]))
        else:
            s_geom = np.cross(n, np.array([1, 0, 0]))
        s_geom = s_geom / np.linalg.norm(s_geom)

    return material.get_cohesion(n, s_geom)


# ==========================================
# 1. КРИТЕРИЙ СДВИГА
# ==========================================
def calculate_criterion_shear(params, stress_tensor, material: mt.Material, mode='3D'):
    n = get_normal_vector(params, mode)
    sigma_n = stress_tensor.normal_stress(n)
    tau_n = stress_tensor.shear_stress_magnitude(n)

    # Используем новую отдельную функцию вычисления сцепления
    C_val = get_cohesion_limit(n, stress_tensor, material)

    # Сопротивление сдвигу
    if sigma_n > 0:
        resistance = C_val + material.mu_tensile * sigma_n
    else:
        resistance = C_val + material.mu * sigma_n

    f_val = tau_n - resistance
    return f_val, n, tau_n, resistance


def find_critical_plane_shear(stress_tensor, material, mode='3D', num_planes=10):
    best_params = None
    max_f = -float('inf')

    # Дискретный поиск по заданному количеству площадок
    params_list = generate_discrete_params(num_planes, mode)
    for p in params_list:
        f_val, _, _, _ = calculate_criterion_shear(p, stress_tensor, material, mode)
        if f_val > max_f:
            max_f = f_val
            best_params = p

    # Полный расчет для наихудшей (критической) площадки
    _, best_n, tau_n, resistance = calculate_criterion_shear(best_params, stress_tensor, material, mode)

    if abs(resistance) < 1e-9:
        utilization = float('inf') if tau_n > 1e-9 else 0.0
    else:
        utilization = tau_n / resistance

    return max_f, best_n, utilization


# ==========================================
# 2. КРИТЕРИЙ РАСТЯЖЕНИЯ
# ==========================================
def calculate_criterion_tensile(params, stress_tensor, material, mode='3D'):
    n = get_normal_vector(params, mode)
    sigma_n = stress_tensor.normal_stress(n)

    # Используем новую отдельную функцию вычисления предела растяжения
    Rp_n = get_tensile_limit(n, material)

    f_val = sigma_n - Rp_n
    return f_val, n, sigma_n, Rp_n


def find_critical_plane_tensile(stress_tensor, material, mode='3D', num_planes=10):
    best_params = None
    max_f = -float('inf')

    # Дискретный поиск
    params_list = generate_discrete_params(num_planes, mode)
    for p in params_list:
        f_val, _, _, _ = calculate_criterion_tensile(p, stress_tensor, material, mode)
        if f_val > max_f:
            max_f = f_val
            best_params = p

    _, best_n, sigma_n, Rp_n = calculate_criterion_tensile(best_params, stress_tensor, material, mode)

    if Rp_n < 1e-9:
        utilization = float('inf')
    else:
        utilization = sigma_n / Rp_n if sigma_n > 0 else 0.0

    return max_f, best_n, utilization


# ==========================================
# 3. КРИТЕРИЙ СЖАТИЯ
# ==========================================
def calculate_criterion_compression(params, stress_tensor, material, mode='3D'):
    n = get_normal_vector(params, mode)
    sigma_n = stress_tensor.normal_stress(n)

    # Используем новую отдельную функцию вычисления предела сжатия
    Rc_n = get_compression_limit(n, material)

    f_val = -sigma_n - Rc_n
    return f_val, n, sigma_n, Rc_n


def find_critical_plane_compression(stress_tensor, material, mode='3D', num_planes=10):
    best_params = None
    max_f = -float('inf')

    # Дискретный поиск
    params_list = generate_discrete_params(num_planes, mode)
    for p in params_list:
        f_val, _, _, _ = calculate_criterion_compression(p, stress_tensor, material, mode)
        if f_val > max_f:
            max_f = f_val
            best_params = p

    _, best_n, sigma_n, Rc_n = calculate_criterion_compression(best_params, stress_tensor, material, mode)

    comp_stress = -sigma_n
    if Rc_n < 1e-9:
        utilization = float('inf')
    else:
        utilization = comp_stress / Rc_n if comp_stress > 0 else 0.0

    return max_f, best_n, utilization
