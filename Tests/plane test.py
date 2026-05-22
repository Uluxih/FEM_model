import numpy as np
import matplotlib.pyplot as plt

# --- Импорты ваших реальных классов ---
from FEM.Abstract.Integration_Point_Level import Material
from FEM.Integration_Point_Level.UbiquitousJointModel3D import UbiquitousJointModel3D
import FEM.Integration_Point_Level.CriticalPlane.material as cp_mt


# 1. Обертка материала (в точности как в вашем основном коде)
class JointedMaterial(Material):
    """Обертка для передачи параметров в ConstitutiveModel"""

    def __init__(self, E, nu, joint_params):
        super().__init__(E, nu)
        self.joint_params = joint_params


def run_rigorous_ip_test():
    print("=== ТЕСТ КОНСТИТУТИВНОЙ МОДЕЛИ НА 1 ТОЧКЕ ИНТЕГРИРОВАНИЯ ===")

    # 2. Инициализация тензора (строгий парсинг)
    tensor_data_str = """
    0.001 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
    0.000000 0.001 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
    0.000000 0.000000 0.001 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
    0.000000 0.000000 0.000000 0.001 0.000000 0.000000 0.000000 0.000000 0.000000
    0.000000 0.000000 0.000000 0.000000 0.001 0.000000 0.000000 0.000000 0.000000
    0.000000 0.000000 0.000000 0.000000 0.000000 0.001 0.000000 0.000000 0.000000
    0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.001 0.000000 0.000000
    0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.001 0.000000
    0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.001
    """
    A_matrix = cp_mt.load_tensor_from_string(tensor_data_str)
    A_matrix=A_matrix*100000000000000*2
    # ВНИМАНИЕ: умножение на 1e18 убрано, так как оно ломает критерий разрушения.

    # 3. Физико-механические свойства (приведены к реалистичным значениям для скалы)
    ROCK_E = 01.50e7   # Модуль Юнга (Па)
    ROCK_NU = 0.2     # Коэффициент Пуассона

    # Прочность матрицы (для поиска критической плоскости)
    ROCK_MU = 0.5     # Коэффициент трения матрицы (поиск)
    ROCK_RP = 15.5e100   # Предел на растяжение (Па)
    ROCK_RC = 15.0e100  # Предел на сжатие (Па)

    cp_material = cp_mt.Material(
        mu=ROCK_MU,
        A_tensor=A_matrix,
        Rpx=ROCK_RP, Rpy=ROCK_RP, Rpz=ROCK_RP,
        Rcx=ROCK_RC, Rcy=ROCK_RC, Rcz=ROCK_RC
    )

    joint_params = {
        'cp_material': cp_material,  # Передаем материал поиска

        # Податливость (жесткость) образующейся трещины
        'kn': 1,  # Нормальная жесткость (Па/м)
        'ks': 1.5e100,  # Сдвиговая жесткость (Па/м)
        'kt': 1.5e100,  # Сдвиговая жесткость (Па/м)
        'spacing': 0.2,  # Расстояние между трещинами (м)

        # Прочность скольжения по образовавшейся трещине
        # ВНИМАНИЕ: Очень низкие значения приведут к резкому сбросу силы (хрупкое разрушение)
        'c': 0.001*100000000*2,  # Сцепление трещины (Па)
        'phi': 0.0,  # Угол внутреннего трения трещины (град)
        'psi': 0.0,  # Угол дилатансии трещины (град)
        't': 15.e10  # Прочность на отрыв по трещине (Па)
    }

    global_material = JointedMaterial(E=ROCK_E, nu=ROCK_NU, joint_params=joint_params)

    # 4. Инициализация вашей конститутивной модели
    model = UbiquitousJointModel3D(global_material)

    # 5. Настройка нагружения (Кинематическое нагружение - чистый сдвиг)
    NUM_STEPS = 350
    MAX_STRAIN = 0.08  # 0.4% деформации (достаточно для разрушения)

    strains_history = []
    stresses_history = []

    lock_strain = None
    lock_stress = None

    print("Запуск шагов деформирования...")
    for step in range(NUM_STEPS + 1):
        # Постепенно увеличиваем сдвиговую деформацию gamma_xy
        gamma_xy = (step / NUM_STEPS) * MAX_STRAIN

        # Вектор Фойгта: [eps_xx, eps_yy, eps_zz, gamma_xy, gamma_yz, gamma_xz]
        current_strain = np.array([0.0, 0.0, 0.0, gamma_xy, 0.0, 0.0])

        # --- ЯДРО МКЭ ---
        stress, D_tangent = model.update_state(current_strain)
        model.commit()
        # ----------------

        # Сохраняем данные (индекс 3 - это сдвиг в плоскости XY)
        tau_xy = stress[3]
        strains_history.append(gamma_xy)
        stresses_history.append(tau_xy)

        # Отлавливаем момент фиксации трещины
        if model.is_locked and lock_strain is None:
            lock_strain = gamma_xy
            lock_stress = tau_xy
            print(f" [!] Сработал критерий разрушения на шаге {step}.")
            print(f"     Деформация: {gamma_xy * 100:.3f}% | Напряжение: {tau_xy / 1e6:.2f} МПа")

    # 6. Построение графика
    strains_percent = np.array(strains_history) * 100  # В проценты для красоты
    stresses_mpa = np.array(stresses_history) / 1e6  # В мегапаскали

    plt.figure(figsize=(10, 6))
    plt.plot(strains_percent, stresses_mpa, color='#1f77b4', linewidth=2.5, label=r'Касательное напряжение $\tau_{xy}$')

    if lock_strain is not None:
        plt.axvline(x=lock_strain * 100, color='red', linestyle='--', linewidth=1.5, label='Момент образования трещины')
        plt.scatter([lock_strain * 100], [lock_stress / 1e6], color='red', s=60, zorder=5)

    plt.title("Поведение материала в одной точке (UbiquitousJointModel3D)", fontsize=14, pad=15)
    plt.xlabel(r"Сдвиговая деформация $\gamma_{xy}$ (%)", fontsize=12)
    plt.ylabel(r"Напряжение сдвига $\tau_{xy}$ (МПа)", fontsize=12)
    plt.grid(True, which='major', linestyle='-', alpha=0.5)
    plt.grid(True, which='minor', linestyle=':', alpha=0.5)
    plt.minorticks_on()
    plt.legend(fontsize=11, loc='lower right')
    plt.tight_layout()

    print("Расчет завершен. Отрисовка графика...")
    plt.show()


if __name__ == "__main__":
    run_rigorous_ip_test()
