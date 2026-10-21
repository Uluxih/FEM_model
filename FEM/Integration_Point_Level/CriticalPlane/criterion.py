import numpy as np
from functools import lru_cache

from FEM.Integration_Point_Level.CriticalPlane.tensor import StressTensor
import FEM.Integration_Point_Level.CriticalPlane.material as mt


# ==========================================
# Функция вычисления вектора нормали
# ==========================================
def get_normal_vector(params, mode='2D_XY'):
    """
    Преобразует угловые параметры в единичный вектор нормали.
    В 3D: params = [theta, phi] (полярный и азимутальный углы)
    В 2D_XY: params = theta (угол в плоскости XY относительно оси X)
    """
    if mode == '3D':
        theta, phi = params[0], params[1]
        nx = np.sin(theta) * np.cos(phi)
        ny = np.sin(theta) * np.sin(phi)
        nz = np.cos(theta)
        return np.array([nx, ny, nz])
    elif mode == '2D_XY':
        # Обработка случая, когда params передается как список [theta] или как число theta
        theta = params[0] if isinstance(params, (list, tuple, np.ndarray)) else params
        nx = np.cos(theta)
        ny = np.sin(theta)
        nz = 0.0
        return np.array([nx, ny, nz])
    else:
        raise ValueError(f"Неизвестный режим: {mode}")


# ==========================================
# ГЕНЕРАЦИЯ ДИСКРЕТНЫХ ПЛОЩАДОК
# ==========================================
def generate_discrete_params(num_planes, mode='2D_XY'):
    """
    Генерирует равномерно распределенный набор параметров (углов) для поиска.
    Для 3D используется алгоритм сферы Фибоначчи для равномерного покрытия сферы.
    Для 2D_XY - равномерное распределение по полуокружности.
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

    elif mode == '2D_XY':
        for i in range(num_planes):
            theta = i * np.pi / num_planes
            params_list.append([theta])
    else:
        raise ValueError(f"Неизвестный режим: {mode}")

    return params_list


@lru_cache(maxsize=64)
def _normals_matrix(num_planes, mode):
    """
    Кэшированная матрица нормалей формы (num_planes, 3).
    Полностью соответствует generate_discrete_params, но строится один раз
    и используется для векторизованного поиска по всем площадкам сразу.
    """
    if mode == '3D':
        if num_planes == 1:
            return np.array([[0.0, 0.0, 1.0]])
        i = np.arange(num_planes, dtype=float)
        y = np.clip(1.0 - (i / (num_planes - 1)) * 2.0, -1.0, 1.0)
        theta = np.arccos(y)
        phi = (i * (np.pi * (3.0 - np.sqrt(5.0)))) % (2.0 * np.pi)
        sin_t = np.sin(theta)
        return np.column_stack((sin_t * np.cos(phi), sin_t * np.sin(phi), np.cos(theta)))
    elif mode == '2D_XY':
        theta = np.arange(num_planes, dtype=float) * (np.pi / num_planes)
        return np.column_stack((np.cos(theta), np.sin(theta), np.zeros(num_planes)))
    else:
        raise ValueError(f"Неизвестный режим: {mode}")


# ==========================================
# ВЕКТОРИЗОВАННЫЕ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================
def _traction_batch(matrix, N):
    """Векторы напряжений t = sigma · n для всех нормалей сразу. N: (m, 3)."""
    # Матрица тензора симметрична, поэтому N @ matrix == (matrix @ N.T).T
    return N @ matrix


def _cohesion_batch(N, T, sigma_n, A_tensor):
    """
    Векторизованный аналог get_cohesion_limit: предельное сцепление C
    для всех площадок одновременно.

    N : (m, 3) — нормали; T : (m, 3) — traction (sigma·n);
    sigma_n : (m,) — нормальные напряжения.
    Возвращает (C : (m,), tau_n : (m,)).
    """
    tau_vec = T - sigma_n[:, None] * N
    tau_n = np.sqrt(np.maximum(np.einsum('ni,ni->n', tau_vec, tau_vec), 0.0))

    # Направление сдвига s_geom
    S = np.empty_like(N)
    big = tau_n > 1e-9
    S[big] = tau_vec[big] / tau_n[big, None]
    if (~big).any():
        # Произвольное направление, ортогональное нормали (как в скалярной версии)
        Ns = N[~big]
        ref = np.where(np.abs(Ns[:, 2:3]) < 0.9,
                       np.array([0.0, 0.0, 1.0]),
                       np.array([1.0, 0.0, 0.0]))
        s = np.cross(Ns, ref)
        s /= np.linalg.norm(s, axis=1, keepdims=True)
        S[~big] = s

    # v = n (x) s, развернутый в вектор длины 9; C = sqrt(v @ A @ v)
    V = (N[:, :, None] * S[:, None, :]).reshape(len(N), 9)
    C_sq = np.einsum('ni,ij,nj->n', V, A_tensor, V)
    return np.sqrt(np.maximum(C_sq, 0.0)), tau_n


# ==========================================
# ОТДЕЛЬНЫЕ ФУНКЦИИ ПРЕДЕЛОВ ПРОЧНОСТИ
# ==========================================
def get_tensile_limit(n, material: mt.Material):
    """Вычисляет предел прочности на растяжение на заданной площадке"""
    return material.Rpx * n[0] ** 2 + material.Rpy * n[1] ** 2 + material.Rpz * n[2] ** 2


def get_compression_limit(n, material: mt.Material):
    """Вычисляет предел прочности на сжатие на заданной площадке"""
    return material.Rcx * n[0] ** 2 + material.Rcy * n[1] ** 2 + material.Rcz * n[2] ** 2


def get_cohesion_limit(n, stress_tensor: StressTensor, material: mt.Material):
    """Вычисляет предельное сцепление (C) на площадке с учетом направления сдвига"""
    # Считаем traction один раз и выводим из него все остальное
    t = stress_tensor.traction(n)
    sigma_n = np.dot(n, t)
    tau_vec = t - sigma_n * n
    tau_n = np.linalg.norm(tau_vec)

    # Направление сдвига (s_geom) – единичный вектор в плоскости площадки
    if tau_n > 1e-9:
        s_geom = tau_vec / tau_n
    else:
        # Выбираем произвольное направление, ортогональное нормали
        # Для плоскости XY n[2] всегда 0, так что это условие отработает идеально
        if abs(n[2]) < 0.9:
            s_geom = np.cross(n, np.array([0, 0, 1]))
        else:
            s_geom = np.cross(n, np.array([1, 0, 0]))
        s_geom = s_geom / np.linalg.norm(s_geom)

    return material.get_cohesion(n, s_geom)


# ==========================================
# 1. КРИТЕРИЙ СДВИГА
# ==========================================
def calculate_criterion_shear(params, stress_tensor, material: mt.Material, mode='2D_XY'):
    n = get_normal_vector(params, mode)

    # Один traction — далее все величины выводятся из него без повторных умножений
    t = stress_tensor.traction(n)
    sigma_n = np.dot(n, t)
    tau_vec = t - sigma_n * n
    tau_n = np.linalg.norm(tau_vec)

    # Сцепление на площадке с учетом направления сдвига
    if tau_n > 1e-9:
        s_geom = tau_vec / tau_n
    else:
        if abs(n[2]) < 0.9:
            s_geom = np.cross(n, np.array([0, 0, 1]))
        else:
            s_geom = np.cross(n, np.array([1, 0, 0]))
        s_geom = s_geom / np.linalg.norm(s_geom)
    C_val = material.get_cohesion(n, s_geom)

    # Сопротивление сдвигу (закон Кулона-Мора)
    # Сжатие - отрицательно (sigma_n < 0), поэтому мы ВЫЧИТАЕМ sigma_n, чтобы сопротивление РОСЛО
    if sigma_n > 0:
        # При растяжении сопротивление падает. Ограничиваем нулем, чтобы не получить отрицательный предел
        resistance = max(0.0, C_val - material.mu * sigma_n)
    else:
        # При сжатии (sigma_n <= 0) сопротивление растет
        resistance = C_val - material.mu * sigma_n

    f_val = tau_n - resistance
    return f_val, n, tau_n, resistance


def find_critical_plane_shear(stress_tensor, material, mode='2D_XY', num_planes=10):
    # Векторизованный расчет критерия сразу для всех площадок
    N = _normals_matrix(num_planes, mode)
    T = _traction_batch(stress_tensor.matrix, N)
    sigma_n = np.einsum('ni,ni->n', N, T)
    C_val, tau_n = _cohesion_batch(N, T, sigma_n, material.A_tensor)

    mu = material.mu
    resistance = np.where(sigma_n > 0.0,
                          np.maximum(0.0, C_val - mu * sigma_n),
                          C_val - mu * sigma_n)
    f = tau_n - resistance

    idx = int(np.argmax(f))
    max_f = float(f[idx])
    best_n = N[idx].copy()
    tau_best = float(tau_n[idx])
    res_best = float(resistance[idx])

    if abs(res_best) < 1e-9:
        utilization = float('inf') if tau_best > 1e-9 else 0.0
    else:
        utilization = tau_best / res_best

    return max_f, best_n, utilization


# ==========================================
# 2. КРИТЕРИЙ РАСТЯЖЕНИЯ
# ==========================================
def calculate_criterion_tensile(params, stress_tensor, material, mode='2D_XY'):
    n = get_normal_vector(params, mode)
    sigma_n = stress_tensor.normal_stress(n)

    # Используем новую отдельную функцию вычисления предела растяжения
    Rp_n = get_tensile_limit(n, material)

    f_val = sigma_n - Rp_n
    return f_val, n, sigma_n, Rp_n


def find_critical_plane_tensile(stress_tensor, material, mode='2D_XY', num_planes=10):
    # Векторизованный расчет критерия сразу для всех площадок
    N = _normals_matrix(num_planes, mode)
    T = _traction_batch(stress_tensor.matrix, N)
    sigma_n = np.einsum('ni,ni->n', N, T)
    Rp = (N * N) @ np.array([material.Rpx, material.Rpy, material.Rpz])

    f = sigma_n - Rp

    idx = int(np.argmax(f))
    max_f = float(f[idx])
    best_n = N[idx].copy()
    s = float(sigma_n[idx])
    r = float(Rp[idx])

    if r < 1e-9:
        utilization = float('inf')
    else:
        utilization = s / r if s > 0 else 0.0

    return max_f, best_n, utilization


# ==========================================
# 3. КРИТЕРИЙ СЖАТИЯ
# ==========================================
def calculate_criterion_compression(params, stress_tensor, material, mode='2D_XY'):
    n = get_normal_vector(params, mode)
    sigma_n = stress_tensor.normal_stress(n)

    # Используем новую отдельную функцию вычисления предела сжатия
    Rc_n = get_compression_limit(n, material)

    f_val = -sigma_n - Rc_n
    return f_val, n, sigma_n, Rc_n


def find_critical_plane_compression(stress_tensor, material, mode='2D_XY', num_planes=10):
    # Векторизованный расчет критерия сразу для всех площадок
    N = _normals_matrix(num_planes, mode)
    T = _traction_batch(stress_tensor.matrix, N)
    sigma_n = np.einsum('ni,ni->n', N, T)
    Rc = (N * N) @ np.array([material.Rcx, material.Rcy, material.Rcz])

    f = -sigma_n - Rc

    idx = int(np.argmax(f))
    max_f = float(f[idx])
    best_n = N[idx].copy()
    comp_stress = -float(sigma_n[idx])
    r = float(Rc[idx])

    if r < 1e-9:
        utilization = float('inf')
    else:
        utilization = comp_stress / r if comp_stress > 0 else 0.0

    return max_f, best_n, utilization
