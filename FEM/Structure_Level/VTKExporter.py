import os
import numpy as np
import pyvista as pv


class VTKExporter:
    """Класс для экспорта результатов МКЭ в формат VTK через PyVista"""

    @staticmethod
    def export(model, filename="results.vtu"):  # Рекомендуется использовать .vtu
        print(f"Экспорт результатов в {filename}...")

        num_nodes = len(model.nodes)
        num_elements = len(model.elements)

        if num_nodes == 0 or num_elements == 0:
            print("Ошибка: Модель пуста. Экспорт отменен.")
            return

        # 1. Маппинг узлов
        node_to_index = {node.id: idx for idx, node in enumerate(model.nodes)}

        # 2. Подготовка координат узлов (Points)
        # Создаем numpy массив (N, 3), заполненный нулями
        points = np.zeros((num_nodes, 3))
        for idx, node in enumerate(model.nodes):
            coords = node.coords
            points[idx, 0] = coords[0] if len(coords) > 0 else 0.0
            points[idx, 1] = coords[1] if len(coords) > 1 else 0.0
            points[idx, 2] = coords[2] if len(coords) > 2 else 0.0

        # 3. Подготовка топологии ячеек (Cells) и их типов
        cells = []
        cell_types = []

        for el in model.elements:
            n_nodes = len(el.nodes)
            indices = [node_to_index[node.id] for node in el.nodes]

            # Формат PyVista для cells: [количество_узлов, индекс_1, индекс_2, ...]
            cells.append(n_nodes)
            cells.extend(indices)

            # Определение типа элемента согласно вашей логике
            if n_nodes == 8:
                cell_types.append(pv.CellType.HEXAHEDRON)  # Эквивалент 12
            elif n_nodes == 4:
                # В вашем коде 4 узла = тип 9 (VTK_QUAD).
                # Если это 3D-тетраэдр, лучше использовать pv.CellType.TETRA (тип 10)
                cell_types.append(pv.CellType.QUAD)
            else:
                cell_types.append(pv.CellType.VERTEX)  # Эквивалент 1

        # Преобразуем списки в numpy массивы для скорости
        cells = np.array(cells)
        cell_types = np.array(cell_types)

        # 4. Создание объекта UnstructuredGrid
        grid = pv.UnstructuredGrid(cells, cell_types, points)

        # 5. Добавление узловых результатов (Point Data - Перемещения)
        displacements = np.zeros((num_nodes, 3))
        for idx, node in enumerate(model.nodes):
            disp = node.displacements
            displacements[idx, 0] = disp[0] if len(disp) > 0 else 0.0
            displacements[idx, 1] = disp[1] if len(disp) > 1 else 0.0
            displacements[idx, 2] = disp[2] if len(disp) > 2 else 0.0

        # Записываем данные в сетку (одной строкой!)
        grid.point_data["Displacements"] = displacements

        # 6. Добавление данных элементов (Cell Data - Нормали)
        normals = np.zeros((num_elements, 3))
        for idx, el in enumerate(model.elements):
            if len(el.integration_points) > 0:
                c_model = el.integration_points[0].constitutive_model
                if hasattr(c_model, 'is_locked') and c_model.is_locked:
                    if c_model.fixed_normal is not None:
                        normals[idx] = c_model.fixed_normal

        grid.cell_data["JointNormal"] = normals

        # 7. Сохранение файла
        # PyVista сама определит нужный формат записи (.vtu или .vtk) по расширению
        grid.save(filename)

        print("Готово!")
