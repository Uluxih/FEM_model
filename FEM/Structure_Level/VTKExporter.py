import os


class VTKExporter:
    """Класс для экспорта результатов МКЭ в формат Legacy VTK (.vtk)"""

    @staticmethod
    def export(model, filename="results.vtk"):
        print(f"Экспорт результатов в {filename}...")

        num_nodes = len(model.nodes)
        num_elements = len(model.elements)

        if num_nodes == 0 or num_elements == 0:
            print("Ошибка: Модель пуста. Экспорт отменен.")
            return

        # Создаем словарь для маппинга ID узлов в последовательные индексы (0, 1, 2...)
        node_to_index = {node.id: idx for idx, node in enumerate(model.nodes)}

        with open(filename, "w", encoding="utf-8") as f:
            # 1. Заголовок VTK файла
            f.write("# vtk DataFile Version 3.0\n")
            f.write("FEM 3D Results\n")
            f.write("ASCII\n")
            f.write("DATASET UNSTRUCTURED_GRID\n\n")

            # 2. Запись координат узлов (POINTS)
            f.write(f"POINTS {num_nodes} float\n")
            for node in model.nodes:
                coords = node.coords
                x = coords[0] if len(coords) > 0 else 0.0
                y = coords[1] if len(coords) > 1 else 0.0
                z = coords[2] if len(coords) > 2 else 0.0
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

            # 3. Запись топологии элементов (CELLS)
            total_list_size = sum(len(el.nodes) + 1 for el in model.elements)
            f.write(f"\nCELLS {num_elements} {total_list_size}\n")
            for el in model.elements:
                n_nodes = len(el.nodes)
                indices = [str(node_to_index[node.id]) for node in el.nodes]
                f.write(f"{n_nodes} " + " ".join(indices) + "\n")

            # 4. Запись типов элементов (CELL_TYPES)
            f.write(f"\nCELL_TYPES {num_elements}\n")
            for el in model.elements:
                n_nodes = len(el.nodes)
                if n_nodes == 8:
                    vtk_type = 12
                elif n_nodes == 4:
                    vtk_type = 9
                else:
                    vtk_type = 1
                f.write(f"{vtk_type}\n")

            # 5. Запись узловых результатов (POINT_DATA)
            f.write(f"\nPOINT_DATA {num_nodes}\n")
            f.write("VECTORS Displacements float\n")
            for node in model.nodes:
                disp = node.displacements
                u = disp[0] if len(disp) > 0 else 0.0
                v = disp[1] if len(disp) > 1 else 0.0
                w = disp[2] if len(disp) > 2 else 0.0
                f.write(f"{u:.6e} {v:.6e} {w:.6e}\n")

            # 6. Запись данных элементов (CELL_DATA)
            f.write(f"\nCELL_DATA {num_elements}\n")

            # Векторное поле: Нормаль трещины (Joint Normal)
            f.write("VECTORS JointNormal float\n")
            for el in model.elements:
                nx, ny, nz = 0.0, 0.0, 0.0
                if len(el.integration_points) > 0:
                    c_model = el.integration_points[0].constitutive_model
                    # ИСПРАВЛЕНИЕ: Читаем нормаль из зафиксированного состояния модели
                    if hasattr(c_model, 'is_locked') and c_model.is_locked:
                        if c_model.fixed_normal is not None:
                            nx, ny, nz = c_model.fixed_normal
                f.write(f"{nx:.6f} {ny:.6f} {nz:.6f}\n")

        print("Готово!")
