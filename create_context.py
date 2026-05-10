import os

# Настройки
OUTPUT_FILE = "project_context.txt"
EXTENSIONS = {".py"}  # Какие файлы собирать
IGNORE_DIRS = {".git", "__pycache__", ".venv", "venv", ".idea", ".vscode"}  # Что игнорировать


def create_context():
    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        # Пишем мета-информацию
        outfile.write("Project Structure and Codebase:\n\n")

        for root, dirs, files in os.walk("."):
            # Удаляем игнорируемые папки из списка обхода
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

            for file in files:
                if any(file.endswith(ext) for ext in EXTENSIONS):
                    file_path = os.path.join(root, file)

                    # Записываем имя файла как разделитель
                    outfile.write(f"\n{'=' * 20}\n")
                    outfile.write(f"File: {file_path}\n")
                    outfile.write(f"{'=' * 20}\n\n")

                    # Читаем и записываем содержимое
                    try:
                        with open(file_path, "r", encoding="utf-8") as infile:
                            outfile.write(infile.read())
                            outfile.write("\n")
                    except Exception as e:
                        outfile.write(f"Error reading file: {e}\n")

    print(f"Готово! Файл контекста создан: {OUTPUT_FILE}")


if __name__ == "__main__":
    create_context()
